from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# ─── Enums ──────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DamageSeverity(str, Enum):
    NO_DAMAGE = "no_damage"
    MINOR = "minor"
    MAJOR = "major"
    DESTROYED = "destroyed"

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class EmotionLabel(str, Enum):
    FEAR = "fear"
    ANGER = "anger"
    SADNESS = "sadness"
    JOY = "joy"
    SURPRISE = "surprise"
    DISGUST = "disgust"

class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    GEOJSON = "geojson"
    KML = "kml"

# ─── GeoJSON Primitives ─────────────────────────────────────────────

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any

# ─── API Request / Response Models ──────────────────────────────────

class ImageIngestRequest(BaseModel):
    source: str = Field(..., description="Satellite source name, e.g. Sentinel-2")
    image_url: Optional[str] = None
    capture_time: datetime
    area: GeoJSONGeometry  # GeoJSON Polygon
    pre_event: bool = False

class ImageJobResponse(BaseModel):
    job_id: str
    source: str
    status: JobStatus
    capture_time: datetime
    created_at: datetime
    map_url: Optional[str] = None
    damage_stats: Optional[Dict[str, int]] = None

class DamageOutput(BaseModel):
    job_id: str
    map_url: str
    damage_stats: Dict[str, int]
    polygons: List[Dict[str, Any]]  # GeoJSON FeatureCollection features
    confidence: float

class TweetIngestRequest(BaseModel):
    tweet_id: str
    text: str
    user_location: Optional[str] = None
    geo: Optional[GeoJSONGeometry] = None  # GeoJSON Point
    timestamp: datetime
    language: str = "en"

class TweetRecord(BaseModel):
    id: str
    tweet_id: str
    text: str
    user_location: Optional[str]
    sentiment: SentimentLabel
    emotion: Dict[str, float]
    geo: Optional[GeoJSONGeometry]
    timestamp: datetime
    created_at: datetime

class DocumentIngestRequest(BaseModel):
    source: str = Field(..., description="Source organization, e.g. UN OCHA")
    title: str
    document_type: Literal["guideline", "situation_report", "technical_manual", "other"]
    file_url: Optional[str] = None
    file_bytes: Optional[bytes] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    section: Optional[str]
    page: Optional[int]
    embedding: Optional[List[float]] = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    source: str
    section: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class RAGQueryRequest(BaseModel):
    query: str
    damage_context: Optional[Dict[str, Any]] = None
    top_k: int = 5
    require_citations: bool = True

class RAGCommentary(BaseModel):
    commentary: str
    citations: List[str]
    confidence: float
    retrieved_chunks: List[RetrievedChunk]

class ReportRequest(BaseModel):
    title: str
    region: GeoJSONGeometry
    start_date: datetime
    end_date: datetime
    include_damage: bool = True
    include_sentiment: bool = True
    include_rag: bool = True
    format: ReportFormat = ReportFormat.PDF

class ReportResponse(BaseModel):
    report_id: str
    title: str
    status: JobStatus
    format: ReportFormat
    download_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class DashboardMetrics(BaseModel):
    total_images_processed: int
    total_tweets_analyzed: int
    total_documents_indexed: int
    damage_summary: Dict[str, int]
    sentiment_timeline: List[Dict[str, Any]]
    recent_alerts: List[str]
