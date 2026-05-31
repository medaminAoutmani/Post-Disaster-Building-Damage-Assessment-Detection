from fastapi import APIRouter, HTTPException
from sqlalchemy import select
import uuid

from app.core.db import AsyncSessionLocal
from app.db_models import ImageJob, Tweet, Report
from app.models.schemas import JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.get("/image/{job_id}")
async def get_image_job(job_id: str):
    job_uuid = uuid.UUID(str(job_id))
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ImageJob).where(ImageJob.id == job_uuid))
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": str(job.id),
            "source": job.source,
            "status": job.status,
            "capture_time": job.capture_time,
            "map_url": job.map_url,
            "damage_stats": job.damage_stats,
            "confidence": job.confidence,
            "created_at": job.created_at,
        }

@router.get("/image")
async def list_image_jobs(limit: int = 20, offset: int = 0):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ImageJob).order_by(ImageJob.created_at.desc()).limit(limit).offset(offset))
        jobs = result.scalars().all()
        return [{"job_id": str(j.id), "source": j.source, "status": j.status, "created_at": j.created_at} for j in jobs]
