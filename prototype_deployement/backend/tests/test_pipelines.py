import pytest
import numpy as np
from app.ml.segmentation import get_segmentation_model
from app.ml.sentiment import get_sentiment_analyzer
from app.ml.rag_engine import get_rag_engine

class TestSegmentation:
    def test_mock_inference(self):
        model = get_segmentation_model()
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        features = model.predict(img)
        assert isinstance(features, list)
        if features:
            assert "properties" in features[0]
            assert "severity" in features[0]["properties"]

class TestSentiment:
    def test_analyze(self):
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze("I am terrified of the flooding!")
        assert "sentiment" in result
        assert "emotions" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]

class TestRAG:
    @pytest.mark.asyncio
    async def test_generate_commentary(self):
        engine = get_rag_engine()
        result = await engine.generate_commentary("What should responders do?", top_k=3)
        assert "commentary" in result
        assert "citations" in result
