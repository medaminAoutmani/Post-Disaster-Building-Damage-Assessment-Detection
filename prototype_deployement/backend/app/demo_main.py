import io
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.ml.rag_engine import get_rag_engine
from app.ml.segmentation import get_segmentation_model
from app.ml.sentiment import get_sentiment_analyzer
from app.models.schemas import ReportRequest, RAGQueryRequest, TweetIngestRequest

settings = get_settings()

app = FastAPI(
    title="Post-Disaster Analytics Local Demo",
    description="Local runnable PDA app without Docker, Postgres, MinIO, Qdrant, or GPU dependencies.",
    version="1.0.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next((parent for parent in APP_DIR.parents if (parent / "pdf_rag").exists()), Path("/app"))
PDF_RAG_DIR = Path(settings.RAG_PDF_DIR)
if not PDF_RAG_DIR.is_absolute():
    PDF_RAG_DIR = (PROJECT_ROOT / PDF_RAG_DIR).resolve()
LOCAL_STORAGE = (PROJECT_ROOT / "local_runtime").resolve()
LOCAL_STORAGE.mkdir(exist_ok=True)

image_jobs: List[Dict[str, Any]] = []
tweets: List[Dict[str, Any]] = []
documents: List[Dict[str, Any]] = []
reports: Dict[str, Dict[str, Any]] = {}


def _utcnow() -> datetime:
    return datetime.utcnow()


def _iso(value: datetime) -> str:
    return value.isoformat() + "Z"


def _severity_counts(features: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"no_damage": 0, "minor": 0, "major": 0, "destroyed": 0}
    for feature in features:
        severity = feature.get("properties", {}).get("severity", "major")
        stats[severity] = stats.get(severity, 0) + 1
    return stats


def _sentiment_timeline() -> List[Dict[str, Any]]:
    now = _utcnow()
    timeline = []
    for hour_offset in range(23, -1, -1):
        hour = now - timedelta(hours=hour_offset)
        positive = 0
        negative = 0
        for item in tweets:
            timestamp = item["timestamp"]
            if timestamp.strftime("%Y-%m-%d %H") == hour.strftime("%Y-%m-%d %H"):
                positive += item["sentiment"] == "positive"
                negative += item["sentiment"] == "negative"
        timeline.append({"hour": hour.strftime("%H:00"), "positive": positive, "negative": negative})
    return timeline


def _demo_damage_summary() -> Dict[str, int]:
    summary = {"no_damage": 12, "minor": 8, "major": 5, "destroyed": 2}
    for job in image_jobs:
        for key, value in (job.get("damage_stats") or {}).items():
            summary[key] = summary.get(key, 0) + value
    return summary


@app.get("/")
async def root():
    return {"message": "Post-Disaster Analytics local demo API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pda-backend-local-demo"}


@app.get("/dashboard/metrics")
async def get_metrics():
    return {
        "total_images_processed": len(image_jobs),
        "total_tweets_analyzed": len(tweets),
        "total_documents_indexed": len(documents) + len(list(PDF_RAG_DIR.glob("*.pdf"))),
        "damage_summary": _demo_damage_summary(),
        "sentiment_timeline": _sentiment_timeline(),
        "recent_alerts": [
            "Local demo mode is active: data is stored in memory for this server session.",
            f"{len(list(PDF_RAG_DIR.glob('*.pdf')))} PDFs available for RAG interpretation.",
            "CV and NLP model fallbacks are enabled until full ML dependencies are installed.",
        ],
    }


@app.get("/jobs/image")
async def list_image_jobs(limit: int = 20, offset: int = 0):
    ordered = list(reversed(image_jobs))
    return ordered[offset:offset + limit]


@app.get("/jobs/image/{job_id}")
async def get_image_job(job_id: str):
    for job in image_jobs:
        if job["job_id"] == job_id:
            return job
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/ingest/image")
async def ingest_image(
    background_tasks: BackgroundTasks,
    source: str = Form(...),
    capture_time: datetime = Form(...),
    area: str = Form(...),
    pre_event: bool = Form(False),
    file: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    file_bytes = await file.read()
    object_path = LOCAL_STORAGE / f"{job_id}_{file.filename or 'image.tif'}"
    object_path.write_bytes(file_bytes)

    rng = np.random.default_rng(abs(hash(job_id)) % (2**32))
    sample = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    features = get_segmentation_model().predict(sample)
    stats = _severity_counts(features)

    job = {
        "job_id": job_id,
        "source": source,
        "status": "completed",
        "capture_time": _iso(capture_time),
        "created_at": _iso(_utcnow()),
        "map_url": f"local_runtime/{object_path.name}",
        "damage_stats": stats,
        "confidence": round(float(np.mean([f["properties"]["confidence"] for f in features])), 3),
    }
    image_jobs.append(job)
    return job


@app.post("/ingest/tweet")
async def ingest_tweet(payload: TweetIngestRequest):
    analysis = get_sentiment_analyzer().analyze(payload.text)
    record = {
        "id": str(uuid.uuid4()),
        "record_id": str(uuid.uuid4()),
        "tweet_id": payload.tweet_id,
        "text": payload.text,
        "user_location": payload.user_location,
        "sentiment": analysis["sentiment"],
        "emotion": analysis["emotions"],
        "dominant_emotion": analysis["dominant_emotion"],
        "timestamp": payload.timestamp,
        "created_at": _utcnow(),
    }
    tweets.append(record)
    return {"record_id": record["record_id"], "status": "processed", "analysis": analysis}


@app.post("/ingest/tweets/batch")
async def ingest_tweets_batch(payload: List[TweetIngestRequest]):
    results = []
    for item in payload:
        result = await ingest_tweet(item)
        results.append({"tweet_id": item.tweet_id, **result})
    return {"processed": len(results), "results": results}


@app.post("/ingest/document")
async def ingest_document(
    source: str = Form(...),
    title: str = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...),
):
    doc_id = str(uuid.uuid4())
    PDF_RAG_DIR.mkdir(exist_ok=True)
    safe_name = f"{doc_id}_{Path(file.filename or 'document.pdf').name}"
    target = PDF_RAG_DIR / safe_name
    target.write_bytes(await file.read())

    engine = get_rag_engine()
    engine.indexed_pdf_chunks = engine.index_local_pdf_folder(str(PDF_RAG_DIR))
    documents.append({
        "document_id": doc_id,
        "source": source,
        "title": title,
        "document_type": document_type,
        "file": str(target),
        "created_at": _iso(_utcnow()),
    })
    return {"document_id": doc_id, "status": "indexed", "title": title}


@app.get("/rag/status")
async def rag_status():
    return get_rag_engine().local_pdf_status()


@app.post("/rag/query")
async def rag_query(payload: RAGQueryRequest):
    result = await get_rag_engine().generate_commentary(
        query=payload.query,
        damage_context=payload.damage_context,
        top_k=payload.top_k,
    )
    return result


@app.post("/reports/generate")
async def generate_report(payload: ReportRequest):
    report_id = str(uuid.uuid4())
    report = {
        "report_id": report_id,
        "title": payload.title,
        "status": "completed",
        "format": payload.format,
        "download_url": None,
        "created_at": _iso(_utcnow()),
        "completed_at": _iso(_utcnow()),
    }
    reports[report_id] = report
    return report


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    report = reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
