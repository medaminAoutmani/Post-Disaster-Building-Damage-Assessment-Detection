import os
import uuid
import re
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from shapely.geometry import shape, mapping

from app.core.config import get_settings
from app.core.db import AsyncSessionLocal
from app.db_models import Tweet
from app.ml.sentiment import get_sentiment_analyzer
from app.core.logging import get_logger

logger = get_logger("pipelines.social")
settings = get_settings()

class SocialMediaPipeline:
    def __init__(self):
        self.sentiment = get_sentiment_analyzer()
        # Load spaCy for NER geocoding (optional, lightweight)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
            logger.warning("spaCy model not available; NER geocoding disabled")

    def clean_text(self, text: str) -> str:
        """Normalize tweet text."""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def geocode_text(self, text: str, user_location: Optional[str]) -> Optional[Dict[str, float]]:
        """Extract location mentions and return mock coordinates."""
        if self.nlp:
            doc = self.nlp(text + " " + (user_location or ""))
            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if locations:
                h = self._stable_hash(locations[0])
                return {"lat": 35.0 + (h % 100) / 100.0, "lon": -115.0 + (h % 200) / 100.0}
        if user_location:
            h = self._stable_hash(user_location)
            return {"lat": 36.0 + (h % 50) / 100.0, "lon": -115.5 + (h % 100) / 100.0}
        return None

    def _stable_hash(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % 10000

    async def ingest_tweet(self, tweet_id: str, text: str, timestamp: datetime, user_location: Optional[str] = None, geo: Optional[Dict] = None, language: str = "en") -> str:
        """Clean, analyze sentiment, geocode, and store tweet."""
        record_uuid = uuid.uuid4()
        record_id = str(record_uuid)
        clean = self.clean_text(text)

        # Sentiment analysis
        analysis = self.sentiment.analyze(clean)

        # Geolocation
        point_geo = None
        if geo:
            point_geo = geo
        else:
            coords = self.geocode_text(clean, user_location)
            if coords:
                point_geo = {"type": "Point", "coordinates": [coords["lon"], coords["lat"]]}

        # Store
        async with AsyncSessionLocal() as session:
            tweet = Tweet(
                id=record_uuid,
                tweet_id=tweet_id,
                text=clean,
                user_location=user_location,
                sentiment=analysis["sentiment"],
                emotion=analysis["emotions"],
                geo=f"SRID=4326;{json.dumps(point_geo)}" if point_geo else None,
                timestamp=timestamp,
                language=language,
            )
            session.add(tweet)
            await session.commit()

        logger.info("Tweet ingested", tweet_id=tweet_id, sentiment=analysis["sentiment"], emotion=analysis["dominant_emotion"])
        return record_id
