import re
from typing import Any, Dict, List

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("ml.sentiment")
settings = get_settings()

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - exercised when NLP deps are absent
    pipeline = None


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_model_name = settings.SENTIMENT_MODEL_NAME
        self.emotion_model_name = settings.EMOTION_MODEL_NAME
        self.sentiment_pipe = None
        self.emotion_pipe = None
        self.mode = "lexicon"
        self._load_models()

    def _load_models(self):
        if pipeline is None:
            logger.warning("transformers is not installed; using lexicon sentiment fallback")
            return

        try:
            local_only = not settings.ALLOW_REMOTE_MODEL_DOWNLOADS
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                tokenizer=self.sentiment_model_name,
                device=-1,
                model_kwargs={"local_files_only": local_only},
                tokenizer_kwargs={"local_files_only": local_only},
            )
            self.emotion_pipe = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                tokenizer=self.emotion_model_name,
                device=-1,
                top_k=None,
                model_kwargs={"local_files_only": local_only},
                tokenizer_kwargs={"local_files_only": local_only},
            )
            self.mode = "transformers"
            logger.info("Sentiment models loaded successfully", local_files_only=local_only)
        except Exception as exc:
            logger.warning("Falling back to lexicon sentiment analyzer", error=str(exc))

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Returns sentiment, score, emotion scores, dominant emotion, and model mode.
        """
        clean = self._clean(text)
        if not clean:
            return self._empty_result()
        if self.mode == "transformers":
            return self._analyze_with_transformers(clean)
        return self._analyze_with_lexicon(clean)

    def _analyze_with_transformers(self, clean: str) -> Dict[str, Any]:
        s_result = self.sentiment_pipe(clean)[0]
        label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        sentiment = label_map.get(s_result["label"], s_result["label"]).lower()

        e_results = self.emotion_pipe(clean)[0]
        emotions = {item["label"].lower(): round(float(item["score"]), 4) for item in e_results}
        dominant_emotion = max(emotions, key=emotions.get)

        return {
            "sentiment": sentiment,
            "sentiment_score": round(float(s_result["score"]), 4),
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "model_mode": self.mode,
        }

    def _analyze_with_lexicon(self, clean: str) -> Dict[str, Any]:
        tokens = set(re.findall(r"[a-z']+", clean.lower()))
        negative = tokens & {
            "afraid", "anxious", "bad", "blocked", "collapsed", "damage", "damaged",
            "danger", "dead", "destroyed", "evacuate", "fear", "flood", "flooded",
            "help", "hurt", "lost", "missing", "panic", "scared", "terrified",
            "trapped", "unsafe", "urgent", "worried",
        }
        positive = tokens & {
            "aid", "rescued", "safe", "shelter", "support", "thanks", "volunteer",
            "relief", "recovering", "reopened", "helping",
        }
        anger = tokens & {"angry", "furious", "ignored", "delay", "delayed", "failed"}
        sadness = tokens & {"sad", "lost", "dead", "missing", "grief", "alone"}
        fear = tokens & {"afraid", "anxious", "danger", "panic", "scared", "terrified", "trapped", "unsafe"}
        joy = tokens & {"safe", "rescued", "thanks", "relief", "reopened"}

        score = len(positive) - len(negative)
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        sentiment_score = min(0.99, 0.55 + (abs(score) * 0.12))

        raw_emotions = {
            "fear": 0.15 + min(0.6, len(fear) * 0.2),
            "anger": 0.10 + min(0.5, len(anger) * 0.2),
            "sadness": 0.10 + min(0.5, len(sadness) * 0.2),
            "joy": 0.10 + min(0.5, len(joy) * 0.2),
            "surprise": 0.08,
            "disgust": 0.06,
        }
        total = sum(raw_emotions.values())
        emotions = {label: round(value / total, 4) for label, value in raw_emotions.items()}
        dominant_emotion = max(emotions, key=emotions.get)

        return {
            "sentiment": sentiment,
            "sentiment_score": round(float(sentiment_score), 4),
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "model_mode": self.mode,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "emotions": {
                "fear": 0.0,
                "anger": 0.0,
                "sadness": 0.0,
                "joy": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
            },
            "dominant_emotion": "neutral",
            "model_mode": self.mode,
        }

    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().replace("\n", " ").split())[:512]

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(t) for t in texts]


_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
