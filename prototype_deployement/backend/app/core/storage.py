from minio import Minio
from app.core.config import get_settings
import io
from typing import BinaryIO

settings = get_settings()

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)

def ensure_bucket():
    if not minio_client.bucket_exists(settings.MINIO_BUCKET):
        minio_client.make_bucket(settings.MINIO_BUCKET)

def upload_file(object_name: str, data: BinaryIO, length: int, content_type: str = "application/octet-stream"):
    ensure_bucket()
    minio_client.put_object(
        settings.MINIO_BUCKET,
        object_name,
        data,
        length,
        content_type=content_type,
    )
    return f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"

def get_file(object_name: str) -> bytes:
    response = minio_client.get_object(settings.MINIO_BUCKET, object_name)
    return response.read()

def delete_file(object_name: str):
    minio_client.remove_object(settings.MINIO_BUCKET, object_name)
