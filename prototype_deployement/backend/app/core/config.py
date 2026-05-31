from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://pda:pda_secret@localhost:5432/disaster_analytics"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "disaster_docs"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # MinIO / S3
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "disaster-data"
    MINIO_SECURE: bool = False

    # Auth
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # LLM
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    ALLOW_REMOTE_MODEL_DOWNLOADS: bool = False
    RAG_PDF_DIR: str = "../pdf_rag"
    RAG_AUTO_INDEX_LOCAL_PDFS: bool = True
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 200

    # ML Models
    DAMAGE_MODEL_PATH: str = "../models_CV/week12_convnext_tiny_gated_ce_best.pt.zip"
    SENTIMENT_MODEL_PATH: str = "./model_cache/sentiment_model"
    SENTIMENT_MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"

    # App
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 500

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
