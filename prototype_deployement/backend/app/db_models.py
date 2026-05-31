from sqlalchemy import Column, String, DateTime, Integer, Float, JSON, ForeignKey, Enum, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from geoalchemy2 import Geometry
from sqlalchemy.orm import relationship
from app.core.db import Base
import uuid
from datetime import datetime

class ImageJob(Base):
    __tablename__ = "image_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(100), nullable=False)
    area = Column(Geometry("POLYGON", srid=4326))
    capture_time = Column(DateTime, nullable=False)
    status = Column(String(20), default="pending")
    map_url = Column(String(500))
    damage_stats = Column(JSON)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    damage_outputs = relationship("DamagePrediction", back_populates="job")

class DamagePrediction(Base):
    __tablename__ = "damage_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("image_jobs.id"))
    severity = Column(String(20))
    geometry = Column(Geometry("POLYGON", srid=4326))
    confidence = Column(Float)
    area_sqm = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("ImageJob", back_populates="damage_outputs")

class Tweet(Base):
    __tablename__ = "tweets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tweet_id = Column(String(50), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    user_location = Column(String(200))
    sentiment = Column(String(20))
    emotion = Column(JSON)
    geo = Column(Geometry("POINT", srid=4326))
    timestamp = Column(DateTime, nullable=False, index=True)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(200))
    title = Column(String(500), nullable=False)
    document_type = Column(String(50))
    file_url = Column(String(500))
    total_pages = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    text = Column(Text, nullable=False)
    section = Column(String(200))
    page = Column(Integer)
    vector_id = Column(String(100))  # Qdrant point ID
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    region = Column(Geometry("POLYGON", srid=4326))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    format = Column(String(20))
    status = Column(String(20), default="pending")
    download_url = Column(String(500))
    include_damage = Column(Boolean, default=True)
    include_sentiment = Column(Boolean, default=True)
    include_rag = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
