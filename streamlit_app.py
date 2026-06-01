"""Streamlit app for satellite + social-media disaster situation reports."""

from __future__ import annotations

import csv
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
for path in [SRC_DIR, SRC_DIR / "week11", SRC_DIR / "week12", SRC_DIR / "week13", SRC_DIR / "week14", SRC_DIR / "week15", SRC_DIR / "week16", SRC_DIR / "week17"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from week12_model_backbones import ArcMarginProduct, ObjectDamageRepresentationModel
from week13_topology_features import TOPOLOGY_FEATURE_NAMES, extract_topology_signature_from_images
from week14_zero_shot_text_classifier import DEFAULT_MODEL_NAME, build_zero_shot_social_payload
from week15_fuse_event import build_event
from week16_build_event_documents import build_document
from week17_generate_situation_report import build_prompt, call_ollama, template_report


PREFERRED_CHECKPOINT = (
    ROOT
    / "results"
    / "week12"
    / "week12_convnext_tiny_gated_effective_no_sampler"
    / "checkpoints"
    / "week12_convnext_tiny_gated_ce_best.pt"
)
TOPOLOGY_CONFIG_CANDIDATES = [
    ROOT / "results" / "week13_topology" / "threshold" / "topology_threshold.json",
    ROOT / "results" / "week13" / "week13_topology" / "threshold" / "topology_threshold.json",
    ROOT / "results" / "week15_inputs" / "topology.json",
]

HUMANITARIAN_KEYWORDS = {
    "infrastructure_damage": ["building", "bridge", "road", "power", "utility", "collapse", "collapsed", "damage", "damaged", "crumble"],
    "affected_people": ["people", "families", "residents", "evacuated", "shelter", "homeless", "suffering"],
    "rescue_efforts": ["rescue", "search", "volunteer", "donation", "aid", "relief", "support", "help"],
    "injured_dead": ["dead", "death", "killed", "injured", "casualties", "fatalities"],
    "missing_or_found_people": ["missing", "found", "trapped"],
    "vehicle_damage": ["vehicle", "car", "truck", "bus"],
}

EMOTION_KEYWORDS = {
    "fear": ["terrifying", "scared", "fear", "panic", "danger", "threat"],
    "sadness": ["sad", "dead", "killed", "praying", "suffering", "tragedy", "heartbreaking"],
    "anger": ["angry", "outrage", "furious", "blame", "failed"],
    "hope": ["hope", "rescue", "help", "support", "relief", "safe", "praying"],
}


def inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #090d14;
            --panel: #101722;
            --panel-soft: #151e2b;
            --panel-line: rgba(148, 163, 184, 0.18);
            --text-main: #e5edf7;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #34d399;
            --warn: #f59e0b;
        }
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(56, 189, 248, 0.14), transparent 32rem),
                linear-gradient(180deg, #090d14 0%, #0c111a 45%, #090d14 100%);
            color: var(--text-main);
        }
        [data-testid="stSidebar"] {
            background: #0b1018;
            border-right: 1px solid var(--panel-line);
        }
        [data-testid="stHeader"] {
            background: rgba(9, 13, 20, 0.78);
            backdrop-filter: blur(12px);
        }
        h1, h2, h3 {
            color: var(--text-main);
            letter-spacing: 0;
        }
        .app-hero {
            padding: 1.25rem 0 1.4rem;
            border-bottom: 1px solid var(--panel-line);
            margin-bottom: 1.25rem;
        }
        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            margin-bottom: 0.35rem;
        }
        .app-title {
            font-size: clamp(2rem, 4vw, 3.5rem);
            font-weight: 800;
            line-height: 1;
            margin: 0;
        }
        .app-subtitle {
            color: var(--text-muted);
            max-width: 58rem;
            margin-top: 0.8rem;
            font-size: 1rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.8rem 0 1rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(21, 30, 43, 0.96), rgba(16, 23, 34, 0.96));
            border: 1px solid var(--panel-line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 6.2rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
        }
        .metric-label {
            color: var(--text-muted);
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06rem;
        }
        .metric-value {
            color: var(--text-main);
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.35rem;
        }
        .metric-note {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .section-panel {
            background: rgba(16, 23, 34, 0.78);
            border: 1px solid var(--panel-line);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .report-box {
            background: #0f1722;
            border: 1px solid var(--panel-line);
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            line-height: 1.72;
            color: var(--text-main);
        }
        .report-box p {
            margin: 0 0 1rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.36);
            background: linear-gradient(180deg, #1d4ed8, #1554b7);
            color: #f8fafc;
            font-weight: 700;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(52, 211, 153, 0.7);
            color: #ffffff;
        }
        [data-testid="stMetricValue"] {
            color: var(--text-main);
        }
        @media (max-width: 900px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 560px) {
            .metric-grid { grid-template-columns: 1fr; }
            .metric-value { font-size: 1.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def render_metric_cards(cards: list[tuple[str, str, str]]) -> None:
    pieces = ['<div class="metric-grid">']
    for label, value, note in cards:
        pieces.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div>'
            f'<div class="metric-note">{html.escape(note)}</div>'
            "</div>"
        )
    pieces.append("</div>")
    st.markdown("".join(pieces), unsafe_allow_html=True)


def report_to_html(report: str) -> str:
    blocks = [block.strip() for block in report.strip().split("\n\n") if block.strip()]
    rendered = ['<div class="report-box">']
    for block in blocks:
        safe = html.escape(block)
        if safe.startswith("# "):
            rendered.append(f"<h2>{safe[2:]}</h2>")
        elif safe.startswith("## "):
            rendered.append(f"<h3>{safe[3:]}</h3>")
        else:
            safe_paragraph = safe.replace("\n", "<br>")
            rendered.append(f"<p>{safe_paragraph}</p>")
    rendered.append("</div>")
    return "".join(rendered)


def average_nested_confidence(items: list[dict[str, Any]], keys: list[str]) -> float | None:
    values = []
    for item in items:
        for key in keys:
            if key in item:
                try:
                    values.append(float(item[key]))
                except (TypeError, ValueError):
                    pass
    if not values:
        return None
    return sum(values) / len(values)


def cv_precision_proxy(satellite_payload: dict[str, Any] | None) -> tuple[float | None, str]:
    if not satellite_payload:
        return None, "Waiting for satellite assessment"
    topology = satellite_payload.get("topology_validation", {})
    confidence = satellite_payload.get("confidence")
    if topology:
        agreement = float(topology.get("topology_cnn_agreement_rate", 0.0))
        if confidence is not None:
            return (float(confidence) + agreement) / 2, "Mean of CV confidence and topology agreement"
        return agreement, "Topology agreement proxy"
    if confidence is not None:
        return float(confidence), "Vision-model confidence proxy"
    predictions = satellite_payload.get("predictions", [])
    nested = average_nested_confidence(predictions, ["confidence"])
    return nested, "Mean crop confidence proxy" if nested is not None else "No confidence field found"


def nlp_precision_proxy(social_payload: dict[str, Any] | None) -> tuple[float | None, str]:
    if not social_payload:
        return None, "Waiting for NLP assessment"
    included = social_payload.get("included_tweets", [])
    nested = average_nested_confidence(included, ["emotion_confidence", "disaster_type_confidence", "confidence", "agreement"])
    if nested is not None:
        return nested, "Mean zero-shot/LLM confidence proxy"
    informative = float(social_payload.get("informative_posts", 0))
    classified = float(social_payload.get("classified_posts", informative))
    if classified > 0:
        return informative / classified, "Informative coverage proxy"
    representative = len(social_payload.get("representative_posts", []))
    if representative:
        return 1.0, "Keyword fallback produced representative posts"
    return None, "No confidence field found"


def decode_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {uploaded_file.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray, size: int = 128) -> torch.Tensor:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@st.cache_resource(show_spinner=False)
def load_week12_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "convnext_tiny")
    fusion = checkpoint.get("fusion", "gated")
    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    model = ObjectDamageRepresentationModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arc_head = None
    if checkpoint.get("arc_head_state_dict") is not None:
        arc_head = ArcMarginProduct(
            embedding_dim,
            len(CLASS_NAMES),
            scale=float(checkpoint.get("arcface_scale", 30.0)),
            margin=float(checkpoint.get("arcface_margin", 0.3)),
        )
        arc_head.load_state_dict(checkpoint["arc_head_state_dict"])
        arc_head.eval()
    return model, arc_head


@st.cache_data(show_spinner=False)
def load_topology_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold_json = payload.get("threshold_json")
    if threshold_json and "class_prototypes" not in payload:
        threshold_path = Path(threshold_json)
        if not threshold_path.is_absolute():
            threshold_path = ROOT / threshold_path
        if threshold_path.exists():
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return payload


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def topology_validate_prediction(
    post_image: np.ndarray,
    diff_image: np.ndarray,
    cnn_prediction: str,
    config: dict[str, Any],
    thresholds: int = 16,
) -> dict[str, Any]:
    if "class_prototypes" not in config:
        raise ValueError("Topology config must contain all-class class_prototypes. Refit Week 13 with the all-class prototype script.")
    features = extract_topology_signature_from_images(post_image, diff_image, thresholds=thresholds)
    vector = np.asarray([features[name] for name in TOPOLOGY_FEATURE_NAMES], dtype=np.float32)
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    normalized = (vector - mean) / std

    distances = {}
    class_indices = [int(index) for index in config.get("prototype_class_indices", range(len(CLASS_NAMES)))]
    for class_index in class_indices:
        class_name = CLASS_NAMES[class_index]
        prototype = np.asarray(config["class_prototypes"][class_name], dtype=np.float32)
        distances[class_name] = float(np.linalg.norm(normalized - prototype))
    topology_prediction = min(distances.items(), key=lambda item: item[1])[0]
    nearest = distances[topology_prediction]
    ordered = sorted(distances.values())
    margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
    return {
        "cnn_prediction": cnn_prediction,
        "topology_prediction": topology_prediction,
        "topology_agrees_with_cnn": topology_prediction == cnn_prediction,
        "nearest_distance": nearest,
        "distance_margin": margin,
        "prototype_distances": distances,
    }


@st.cache_resource(show_spinner=False)
def load_zero_shot_classifier(model_name: str, device: str):
    from transformers import pipeline

    if device == "auto":
        pipeline_device = 0 if torch.cuda.is_available() else -1
    elif device == "cpu":
        pipeline_device = -1
    elif device == "cuda":
        pipeline_device = 0
    else:
        pipeline_device = int(device)
    return pipeline("zero-shot-classification", model=model_name, tokenizer=model_name, device=pipeline_device)


@torch.no_grad()
def predict_crop_pairs(
    pre_files,
    post_files,
    checkpoint_path: Path,
    topology_config: dict[str, Any] | None = None,
    topology_thresholds: int = 16,
) -> dict[str, Any]:
    if len(pre_files) != len(post_files):
        raise ValueError("Upload the same number of pre-disaster and post-disaster building crops.")
    if not pre_files:
        raise ValueError("Upload at least one pre/post building crop pair.")

    model, arc_head = load_week12_model(str(checkpoint_path))
    counts: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    total_confidence = 0.0
    rows = []
    topology_rows = []
    topology_agreements = 0

    for pre_file, post_file in zip(pre_files, post_files):
        pre_image = decode_image(pre_file)
        post_image = decode_image(post_file)
        diff_image = np.abs(pre_image.astype(np.float32) - post_image.astype(np.float32)).astype(np.uint8)

        pre = preprocess_image(pre_image).unsqueeze(0)
        post = preprocess_image(post_image).unsqueeze(0)
        diff = preprocess_image(diff_image).unsqueeze(0)
        logits, embeddings = model(pre, post, diff, return_embedding=True)
        if arc_head is not None:
            logits = arc_head.cosine_logits(embeddings)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction].item())
        class_name = CLASS_NAMES[prediction]
        counts[class_name] += 1
        confidence_sum[class_name] += confidence
        total_confidence += confidence
        rows.append({"pre": pre_file.name, "post": post_file.name, "prediction": class_name, "confidence": confidence})
        if topology_config is not None:
            topology_result = topology_validate_prediction(
                post_image,
                diff_image,
                class_name,
                topology_config,
                thresholds=topology_thresholds,
            )
            topology_agreements += int(topology_result["topology_agrees_with_cnn"])
            topology_rows.append(
                {
                    "pre": pre_file.name,
                    "post": post_file.name,
                    **topology_result,
                }
            )

    total = len(rows)
    payload = {
        "source": "streamlit_week12_preferred_convnext_tiny_gated",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "total_buildings": total,
        "no_damage": counts["no_damage"],
        "minor": counts["minor_damage"],
        "major": counts["major_damage"],
        "destroyed": counts["destroyed"],
        "total_damaged": counts["minor_damage"] + counts["major_damage"] + counts["destroyed"],
        "confidence": total_confidence / max(total, 1),
        "class_confidence": {
            class_name: confidence_sum[class_name] / max(counts[class_name], 1)
            for class_name in CLASS_NAMES
            if counts[class_name] > 0
        },
        "predictions": rows,
    }
    if topology_config is not None:
        payload["topology_validation"] = {
            "role": "all_class_damage_classification_validation",
            "validated": True,
            "validated_buildings": len(topology_rows),
            "topology_cnn_agreement_rate": topology_agreements / max(len(topology_rows), 1),
            "class_prototypes": list(topology_config.get("class_prototypes", {}).keys()),
            "predictions": topology_rows,
        }
    return payload


def parse_tweets(text: str, uploaded_file) -> list[str]:
    tweets = [line.strip() for line in text.splitlines() if line.strip()]
    if uploaded_file is not None:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        if uploaded_file.name.lower().endswith(".csv"):
            reader = csv.DictReader(io.StringIO(raw))
            for row in reader:
                value = row.get("tweet_text") or row.get("text") or row.get("tweet") or ""
                if value.strip():
                    tweets.append(value.strip())
        else:
            tweets.extend(line.strip() for line in raw.splitlines() if line.strip())
    return list(dict.fromkeys(tweets))


def count_keyword_matches(tweet: str, keyword_map: dict[str, list[str]]) -> str | None:
    lowered = tweet.lower()
    scores = {
        label: sum(1 for keyword in keywords if keyword in lowered)
        for label, keywords in keyword_map.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score > 0 else None


def aggregate_tweets(tweets: list[str]) -> dict[str, Any]:
    humanitarian = Counter()
    emotion = Counter()
    informative = 0
    for tweet in tweets:
        human_label = count_keyword_matches(tweet, HUMANITARIAN_KEYWORDS)
        emotion_label = count_keyword_matches(tweet, EMOTION_KEYWORDS)
        if human_label is not None:
            humanitarian[human_label] += 1
            informative += 1
        else:
            humanitarian["other_relevant_information"] += 1
            informative += 1
        emotion[emotion_label or "neutral"] += 1
    return {
        "source": "streamlit_keyword_social_aggregation",
        "classified_posts": len(tweets),
        "informative_posts": informative,
        "humanitarian": dict(humanitarian),
        "emotion": dict(emotion),
        "representative_posts": tweets[:5],
    }


def save_social_payload(event_id: str, social_payload: dict[str, Any]) -> Path:
    event_dir = ROOT / "results" / "streamlit_app"
    event_dir.mkdir(parents=True, exist_ok=True)
    social_json = event_dir / f"{event_id}_social_week15_input.json"
    social_json.write_text(json.dumps(social_payload, indent=2), encoding="utf-8")
    return social_json


def save_outputs(event: dict[str, Any], document: dict[str, Any], report: str) -> tuple[Path, Path, Path]:
    event_dir = ROOT / "results" / "streamlit_app"
    event_dir.mkdir(parents=True, exist_ok=True)
    event_json = event_dir / f"{event['event']}.json"
    doc_json = event_dir / f"{event['event']}_document.json"
    report_md = event_dir / f"{event['event']}_report.md"
    event_json.write_text(json.dumps(event, indent=2), encoding="utf-8")
    doc_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
    report_md.write_text(report + "\n", encoding="utf-8")
    return event_json, doc_json, report_md


st.set_page_config(page_title="Disaster Situation Report Generator", layout="wide")
inject_dark_theme()
st.markdown(
    """
    <div class="app-hero">
        <div class="app-kicker">Multimodal Disaster Intelligence</div>
        <h1 class="app-title">Situation Report Generator</h1>
        <div class="app-subtitle">
            Fuse satellite damage detection with crisis-language signals, review model precision indicators,
            and generate a longer human-readable report with local Ollama support.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model Status")
    checkpoint_path = st.text_input("Week 12 vision checkpoint", value=str(PREFERRED_CHECKPOINT.relative_to(ROOT)))
    checkpoint = ROOT / checkpoint_path
    if checkpoint.exists():
        st.success("Preferred Week 12 gated ConvNeXt checkpoint found.")
    else:
        st.warning("Preferred vision checkpoint not found. Use manual satellite counts.")
    default_topology_path = first_existing_path(TOPOLOGY_CONFIG_CANDIDATES)
    topology_path = st.text_input("Week 13 topology prototypes", value=str(default_topology_path.relative_to(ROOT)))
    topology_config_path = ROOT / topology_path
    enable_topology = st.checkbox("Validate uploaded crop predictions with topology", value=topology_config_path.exists())
    topology_thresholds = st.number_input("Topology thresholds", min_value=4, max_value=64, value=16, step=4)
    st.caption(
        "Topology is a validation/agreement signal, not the main classifier. "
        "It is most useful when damage creates visible structural or topological change; "
        "subtle texture-only damage may not be captured. The CNN prediction remains primary."
    )
    if topology_config_path.exists():
        try:
            topology_preview = load_topology_config(str(topology_config_path))
            if "class_prototypes" in topology_preview:
                st.success("Week 13 all-class topology prototypes found.")
            else:
                st.warning("Topology JSON found, but it is the older no/minor format. Refit Week 13 all-class prototypes before enabling validation.")
        except Exception as exc:
            st.warning(f"Could not inspect topology JSON: {exc}")
    else:
        st.warning("Topology prototype JSON not found. Fit Week 13 prototypes or upload topology JSON later.")
    st.caption("Final fusion uses satellite damage assessment + social-media context.")
    st.divider()
    st.subheader("Report LLM")
    use_ollama = st.toggle("Generate report with Ollama", value=True)
    ollama_model = st.text_input("Ollama model", value="llama3")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434/api/generate")
    st.caption("If Ollama is unavailable, the app falls back to the local paragraph report.")

event_id = st.text_input("Event ID", value="validation_demo")

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.header("Satellite Damage")
    mode = st.radio("Input mode", ["Manual counts", "Upload building crop pairs", "Upload satellite JSON"], horizontal=False)
    satellite_payload: dict[str, Any] | None = None

    if mode == "Manual counts":
        destroyed = st.number_input("Destroyed buildings", min_value=0, value=1496, step=1)
        major = st.number_input("Major damage buildings", min_value=0, value=77, step=1)
        minor = st.number_input("Minor damage buildings", min_value=0, value=60, step=1)
        confidence = st.slider("Vision confidence", 0.0, 1.0, 0.91)
        satellite_payload = {"destroyed": destroyed, "major": major, "minor": minor, "confidence": confidence}
    elif mode == "Upload satellite JSON":
        uploaded_json = st.file_uploader("Satellite JSON", type=["json"], key="satellite_json")
        if uploaded_json is not None:
            satellite_payload = json.loads(uploaded_json.getvalue().decode("utf-8"))
    else:
        st.caption("Upload aligned object-level building crops. Full-scene xBD images require a building extraction step before this classifier.")
        pre_files = st.file_uploader("Pre-disaster building crops", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="pre")
        post_files = st.file_uploader("Post-disaster building crops", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="post")
        if st.button("Run Week 12 Vision Model", disabled=not checkpoint.exists()):
            try:
                topology_config = None
                if enable_topology:
                    if not topology_config_path.exists():
                        raise FileNotFoundError(f"Topology prototype JSON not found: {topology_config_path}")
                    topology_config = load_topology_config(str(topology_config_path))
                satellite_payload = predict_crop_pairs(
                    pre_files,
                    post_files,
                    checkpoint,
                    topology_config=topology_config,
                    topology_thresholds=int(topology_thresholds),
                )
                st.session_state["satellite_payload"] = satellite_payload
            except Exception as exc:
                st.error(str(exc))
        satellite_payload = st.session_state.get("satellite_payload")

    if satellite_payload:
        cv_score, cv_note = cv_precision_proxy(satellite_payload)
        render_metric_cards(
            [
                ("CV Precision", pct(cv_score), cv_note),
                ("Damaged Buildings", str(satellite_payload.get("total_damaged", satellite_payload.get("destroyed", 0) + satellite_payload.get("major", 0) + satellite_payload.get("minor", 0))), "Minor + major + destroyed"),
                ("Destroyed", str(satellite_payload.get("destroyed", 0)), "Highest severity class"),
                ("Model Confidence", pct(satellite_payload.get("confidence")), "Mean vision confidence"),
            ]
        )
        with st.expander("Satellite payload JSON"):
            st.json(satellite_payload)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.header("Social Media")
    social_mode = st.radio(
        "Social input mode",
        ["Paste/upload tweets + DeBERTa zero-shot", "Paste/upload tweets + keyword fallback", "Upload social JSON"],
        horizontal=False,
    )
    social_payload: dict[str, Any] | None = None

    if social_mode == "Upload social JSON":
        uploaded_social = st.file_uploader("Social JSON", type=["json"], key="social_json")
        if uploaded_social is not None:
            social_payload = json.loads(uploaded_social.getvalue().decode("utf-8"))
    else:
        tweet_text = st.text_area("Tweets, one per line", height=220)
        tweet_file = st.file_uploader("Optional TXT or CSV with tweet_text column", type=["txt", "csv"], key="tweets")
        tweets = parse_tweets(tweet_text, tweet_file)
        st.caption(f"{len(tweets)} tweets loaded.")
        if social_mode == "Paste/upload tweets + DeBERTa zero-shot":
            deberta_model_name = st.text_input("Zero-shot DeBERTa model", value=DEFAULT_MODEL_NAME)
            deberta_device = st.selectbox("Zero-shot device", ["auto", "cpu", "cuda"], index=0)
            deberta_batch_size = st.number_input("Zero-shot batch size", min_value=1, max_value=32, value=8, step=1)
            if st.button("Classify Included Tweets", disabled=not tweets):
                try:
                    with st.spinner("Classifying included tweets with zero-shot DeBERTa..."):
                        classifier = load_zero_shot_classifier(deberta_model_name, deberta_device)
                        social_payload = build_zero_shot_social_payload(
                            tweets,
                            classifier,
                            event=event_id,
                            split="streamlit",
                            batch_size=int(deberta_batch_size),
                            source=f"streamlit_zero_shot:{deberta_model_name}",
                        )
                    st.session_state["social_payload"] = social_payload
                except Exception as exc:
                    st.error(str(exc))
            social_payload = st.session_state.get("social_payload")
        else:
            social_payload = aggregate_tweets(tweets) if tweets else None

    if social_payload:
        nlp_score, nlp_note = nlp_precision_proxy(social_payload)
        humanitarian_top = sorted_counts(social_payload.get("humanitarian", {}))
        emotion_top = sorted_counts(social_payload.get("emotion", {}))
        render_metric_cards(
            [
                ("NLP Precision", pct(nlp_score), nlp_note),
                ("Posts", str(social_payload.get("classified_posts", social_payload.get("informative_posts", 0))), "Classified social posts"),
                ("Top Topic", humanitarian_top[0][0].replace("_", " ") if humanitarian_top else "N/A", "Dominant humanitarian label"),
                ("Top Emotion", emotion_top[0][0].replace("_", " ") if emotion_top else "N/A", "Dominant sentiment label"),
            ]
        )
        with st.expander("Social payload JSON"):
            st.json(social_payload)
        st.download_button(
            "Download Week 15 social JSON",
            json.dumps(social_payload, indent=2),
            file_name=f"{event_id}_social_week15_input.json",
            mime="application/json",
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.header("Situation Report")
generate = st.button("Generate Report", type="primary", disabled=not (event_id and satellite_payload and social_payload))
if generate and satellite_payload and social_payload:
    social_json = save_social_payload(event_id, social_payload)
    event = build_event(
        event_id,
        satellite_payload,
        social_payload,
        {"satellite_source": mode, "social_source": social_mode, "social_json": str(social_json.relative_to(ROOT))},
    )
    document = build_document(event)
    report = ""
    if use_ollama:
        try:
            with st.spinner(f"Generating a longer report with Ollama ({ollama_model})..."):
                report = call_ollama(build_prompt(event), ollama_model, ollama_url)
        except Exception as exc:
            st.warning(f"Ollama generation failed, using local paragraph report instead: {exc}")
    if not report:
        report = template_report(event)
    event_json, doc_json, report_md = save_outputs(event, document, report)

    tab_report, tab_event, tab_document = st.tabs(["Report", "Fusion JSON", "RAG Document"])
    with tab_report:
        st.markdown(report_to_html(report), unsafe_allow_html=True)
        st.download_button("Download report", report, file_name=f"{event_id}_report.md")
    with tab_event:
        st.json(event)
        st.caption(f"Saved to {event_json.relative_to(ROOT)}")
    with tab_document:
        st.json(document)
        st.caption(f"Saved to {doc_json.relative_to(ROOT)}")
    st.success(f"Saved report to {report_md.relative_to(ROOT)}")
