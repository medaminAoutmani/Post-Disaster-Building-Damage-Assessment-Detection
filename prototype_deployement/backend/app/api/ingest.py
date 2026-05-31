from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional
from datetime import datetime
import json

from app.models.schemas import ImageIngestRequest, TweetIngestRequest, DocumentIngestRequest, ImageJobResponse
from app.pipelines.imagery import ImageryPipeline
from app.pipelines.documents import DocumentPipeline
from app.pipelines.social_media import SocialMediaPipeline
from app.core.logging import get_logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger("api.ingest")

# ─── Imagery Ingestion ──────────────────────────────────────────────

@router.post("/image", response_model=ImageJobResponse)
async def ingest_image(
    background_tasks: BackgroundTasks,
    source: str = Form(...),
    capture_time: datetime = Form(...),
    area: str = Form(...),  # GeoJSON Polygon string
    pre_event: bool = Form(False),
    file: UploadFile = File(...),
):
    """Ingest satellite imagery and trigger damage segmentation."""
    try:
        area_geo = json.loads(area)
        file_data = await file.read()

        pipeline = ImageryPipeline()
        job_id = await pipeline.ingest_image(source, capture_time, area_geo, file_data, file.filename, pre_event)

        # Trigger async processing
        background_tasks.add_task(pipeline.process_job, job_id)

        return ImageJobResponse(
            job_id=job_id,
            source=source,
            status="pending",
            capture_time=capture_time,
            created_at=datetime.utcnow(),
        )
    except Exception as e:
        logger.error("Image ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ─── Document Ingestion ─────────────────────────────────────────────

@router.post("/document")
async def ingest_document(
    source: str = Form(...),
    title: str = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
):
    """Ingest PDF document, extract text, chunk, embed, and index."""
    try:
        file_data = await file.read()
        pipeline = DocumentPipeline()
        doc_id = await pipeline.ingest_document(source, title, document_type, file_data, file.filename)
        return {"document_id": doc_id, "status": "indexed", "title": title}
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ─── Social Media Ingestion ─────────────────────────────────────────

@router.post("/tweet")
async def ingest_tweet(payload: TweetIngestRequest):
    """Ingest a single tweet for sentiment and geolocation analysis."""
    try:
        pipeline = SocialMediaPipeline()
        record_id = await pipeline.ingest_tweet(
            tweet_id=payload.tweet_id,
            text=payload.text,
            timestamp=payload.timestamp,
            user_location=payload.user_location,
            geo=payload.geo.dict() if payload.geo else None,
            language=payload.language,
        )
        return {"record_id": record_id, "status": "processed"}
    except Exception as e:
        logger.error("Tweet ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tweets/batch")
async def ingest_tweets_batch(tweets: list[TweetIngestRequest]):
    """Batch ingest tweets."""
    pipeline = SocialMediaPipeline()
    results = []
    for t in tweets:
        try:
            rid = await pipeline.ingest_tweet(t.tweet_id, t.text, t.timestamp, t.user_location, t.geo.dict() if t.geo else None, t.language)
            results.append({"tweet_id": t.tweet_id, "record_id": rid, "status": "ok"})
        except Exception as e:
            results.append({"tweet_id": t.tweet_id, "status": "error", "detail": str(e)})
    return {"processed": len(results), "results": results}
