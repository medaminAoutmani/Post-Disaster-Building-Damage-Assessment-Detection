from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select
from app.core.db import AsyncSessionLocal
from app.db_models import Report
from app.models.schemas import ReportRequest, ReportResponse, ReportFormat, JobStatus
from app.services.report_generator import get_report_generator
from app.services.gis_export import get_gis_exporter
from app.core.storage import upload_file
from app.core.logging import get_logger
import io
import json
import uuid
from datetime import datetime

router = APIRouter(prefix="/reports", tags=["Reports"])
logger = get_logger("api.reports")

async def _generate_report_task(report_id: str, title: str, region: dict, start_date, end_date, include_damage: bool, include_sentiment: bool, include_rag: bool, fmt: ReportFormat):
    """Background task to generate report."""
    async with AsyncSessionLocal() as session:
        report = await session.get(Report, uuid.UUID(str(report_id)))
        try:
            gen = get_report_generator()

            if fmt == ReportFormat.PDF:
                data = await gen.generate_pdf(report_id, title, region, start_date, end_date, include_damage, include_sentiment, include_rag)
                content_type = "application/pdf"
                ext = "pdf"
            elif fmt == ReportFormat.HTML:
                data = (await gen.generate_html(report_id, title, region, start_date, end_date, include_damage, include_sentiment, include_rag)).encode()
                content_type = "text/html"
                ext = "html"
            else:
                data = b'{"status": "not_implemented"}'
                content_type = "application/json"
                ext = "json"

            object_name = f"reports/{report_id}/report.{ext}"
            upload_file(object_name, io.BytesIO(data), len(data), content_type)

            report.status = "completed"
            report.download_url = object_name
            await session.commit()
            logger.info("Report generated", report_id=report_id, format=fmt)
        except Exception as e:
            report.status = "failed"
            await session.commit()
            logger.error("Report generation failed", report_id=report_id, error=str(e))

@router.post("/generate", response_model=ReportResponse)
async def generate_report(req: ReportRequest, background_tasks: BackgroundTasks):
    """Queue a report generation job."""
    report_uuid = uuid.uuid4()
    report_id = str(report_uuid)

    async with AsyncSessionLocal() as session:
        report = Report(
            id=report_uuid,
            title=req.title,
            region=f"SRID=4326;{json.dumps(req.region.dict())}",
            start_date=req.start_date,
            end_date=req.end_date,
            format=req.format.value,
            status="pending",
            include_damage=req.include_damage,
            include_sentiment=req.include_sentiment,
            include_rag=req.include_rag,
        )
        session.add(report)
        await session.commit()

    background_tasks.add_task(
        _generate_report_task,
        report_id, req.title, req.region.dict(), req.start_date, req.end_date,
        req.include_damage, req.include_sentiment, req.include_rag, req.format
    )

    return ReportResponse(
        report_id=report_id,
        title=req.title,
        status=JobStatus.PENDING,
        format=req.format,
        created_at=datetime.utcnow(),
    )

@router.get("/{report_id}")
async def get_report(report_id: str):
    async with AsyncSessionLocal() as session:
        report = await session.get(Report, uuid.UUID(str(report_id)))
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return {
            "report_id": str(report.id),
            "title": report.title,
            "status": report.status,
            "format": report.format,
            "download_url": report.download_url,
            "created_at": report.created_at,
            "completed_at": report.completed_at,
        }
