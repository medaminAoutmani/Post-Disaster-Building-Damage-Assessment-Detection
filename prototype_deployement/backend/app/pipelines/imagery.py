import os
import io
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import shape, mapping

from app.core.config import get_settings
from app.core.storage import upload_file, get_file
from app.core.db import AsyncSessionLocal
from app.db_models import ImageJob, DamagePrediction
from app.ml.segmentation import get_segmentation_model
from app.core.logging import get_logger

logger = get_logger("pipelines.imagery")
settings = get_settings()

class ImageryPipeline:
    def __init__(self):
        self.segmentation = get_segmentation_model()

    async def ingest_image(self, source: str, capture_time: datetime, area_geojson: dict, file_data: bytes, filename: str, pre_event: bool = False) -> str:
        """Ingest satellite image, store to object storage, create DB job."""
        job_uuid = uuid.uuid4()
        job_id = str(job_uuid)

        # Store raw image
        object_name = f"imagery/{job_id}/{filename}"
        upload_file(object_name, io.BytesIO(file_data), len(file_data), content_type="image/tiff")

        # Create DB record
        async with AsyncSessionLocal() as session:
            job = ImageJob(
                id=job_uuid,
                source=source,
                area=f"SRID=4326;{json.dumps(area_geojson)}",
                capture_time=capture_time,
                status="pending",
            )
            session.add(job)
            await session.commit()

        logger.info("Image ingested", job_id=job_id, source=source)
        return job_id

    async def process_job(self, job_id: str):
        """Run segmentation and store results."""
        job_uuid = uuid.UUID(str(job_id))
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            result = await session.execute(select(ImageJob).where(ImageJob.id == job_uuid))
            job = result.scalar_one_or_none()
            if not job:
                raise ValueError(f"Job {job_id} not found")

            job.status = "processing"
            await session.commit()

            try:
                # In real system, fetch image from storage and preprocess
                # Here we create a mock numpy array representing the image
                mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

                # Run model
                features = self.segmentation.predict(mock_image)

                # Aggregate stats
                stats = {"no_damage": 0, "minor": 0, "major": 0, "destroyed": 0}
                for f in features:
                    sev = f["properties"]["severity"]
                    stats[sev] = stats.get(sev, 0) + 1

                # Store damage mask as GeoJSON
                mask_geojson = {"type": "FeatureCollection", "features": features}
                mask_bytes = json.dumps(mask_geojson).encode()
                mask_object = f"outputs/{job_id}/damage_mask.geojson"
                upload_file(mask_object, io.BytesIO(mask_bytes), len(mask_bytes), "application/geo+json")

                # Update job
                job.status = "completed"
                job.map_url = mask_object
                job.damage_stats = stats
                job.confidence = round(np.mean([f["properties"]["confidence"] for f in features]), 3)

                # Insert damage predictions
                for feat in features:
                    geom = shape(feat["geometry"])
                    pred = DamagePrediction(
                        job_id=job_uuid,
                        severity=feat["properties"]["severity"],
                        geometry=f"SRID=4326;{json.dumps(mapping(geom))}",
                        confidence=feat["properties"]["confidence"],
                        area_sqm=feat["properties"].get("area_sqm", 0),
                    )
                    session.add(pred)

                await session.commit()
                logger.info("Image processing completed", job_id=job_id, stats=stats)

            except Exception as e:
                job.status = "failed"
                await session.commit()
                logger.error("Image processing failed", job_id=job_id, error=str(e))
                raise
