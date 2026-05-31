from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import ingest, jobs, reports, rag, dashboard
from app.core.logging import get_logger

logger = get_logger("main")

app = FastAPI(
    title="Post-Disaster Analytics Platform",
    description="Production-scale platform integrating satellite imagery, documents, and social media for disaster response insights.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(ingest.router)
app.include_router(jobs.router)
app.include_router(reports.router)
app.include_router(rag.router)
app.include_router(dashboard.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pda-backend"}

@app.get("/")
async def root():
    return {
        "message": "Post-Disaster Analytics Platform API",
        "docs": "/docs",
        "version": "1.0.0",
    }

logger.info("Application startup complete")
