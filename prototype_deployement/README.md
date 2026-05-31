# Post-Disaster Analytics Platform (PDA)

[![Documentation Status](https://readthedocs.org/projects/YOUR_PROJECT/badge/?version=latest)](https://YOUR_PROJECT.readthedocs.io/en/latest/)

A production-scale platform that integrates **satellite imagery**, **PDF documents**, and **social media** to produce integrated damage maps, expert commentary, and consolidated reports using ML and RAG.

## Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Satellite  │  │    PDFs     │  │   Tweets    │
│   Imagery   │  │  Documents  │  │  (Twitter)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────┐
│          Ingestion & Preprocessing            │
│   • GDAL/GeoTIFF  • OCR/Chunking  • Kafka   │
└─────────────────────────────────────────────┘
       │                │                │
       ▼                ▼                ▼
┌──────────┐   ┌──────────────┐   ┌──────────┐
│  Imagery │   │  Vector DB   │   │  Tweet   │
│  Store   │   │  (Qdrant)    │   │  Queue   │
│  (MinIO) │   │  Embeddings  │   │  (Redis) │
└────┬─────┘   └──────┬───────┘   └────┬─────┘
     │                │                │
     ▼                ▼                ▼
┌─────────────────────────────────────────────┐
│           ML Inference Services             │
│  • U-Net Segmentation  • Sentiment/Emotion  │
│  • RAG (BM25 + Dense)  • LLM Commentary    │
└─────────────────────────────────────────────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────┐
│         Report Generation & UI              │
│   • PDF (ReportLab)  • Dashboard (React)    │
│   • GIS Exports      • Alerts/Monitoring   │
└─────────────────────────────────────────────┘
```

## Quick Start (Docker Demo)

This is the recommended path for reviewers. It requires Docker Compose or
podman-compose only. No Python or npm setup is needed on the host.

```bash
chmod +x run_docker_demo.sh
./run_docker_demo.sh
```

Open:

```text
http://localhost:3000
```

Check the API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/rag/status
```

Stop:

```bash
docker compose -f docker-compose.demo.yml down
# or
podman-compose -f docker-compose.demo.yml down
```

## Full Stack (Docker Compose)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional; platform works in template mode without it)

# 2. Build and start all services
make build
make up

# 3. Initialize database
docker-compose exec backend python -m app.init_db

# 4. Seed demo data
make seed

# 5. Open interfaces
# Frontend Dashboard: http://localhost:3000
# API Docs (Swagger): http://localhost:8000/docs
# MinIO Console:      http://localhost:9001 (minioadmin / minioadmin)
# Qdrant Dashboard:   http://localhost:6333/dashboard
```

## Quick Start (Local Demo, No Docker)

Use this path first if you just want a working app with your `pdf_rag/` PDFs,
mock CV inference, lexicon sentiment, and local RAG fallback.

One-command launcher:

```bash
chmod +x run_local_demo.sh
./run_local_demo.sh
```

Open:

```text
http://localhost:3000
```

Manual run:

```bash
# Terminal 1: backend
cd backend
python -m pip install -r requirements-demo.txt
PYTHONPATH=. uvicorn app.demo_main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2: frontend
cd frontend
npm install
npm start
```

Open:

```text
http://localhost:3000
```

Check the API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/rag/status
```

## Services

| Service     | Port | Purpose                              |
|-------------|------|--------------------------------------|
| Frontend    | 3000 | React dashboard with maps & charts   |
| Backend API | 8000 | FastAPI: ingestion, inference, RAG   |
| PostgreSQL  | 5432 | Relational + PostGIS metadata store    |
| Qdrant      | 6333 | Vector database for RAG embeddings   |
| MinIO       | 9000 | S3-compatible object storage         |
| Redis       | 6379 | Caching & job queue                  |

## API Endpoints

### Ingestion
- `POST /ingest/image` — Upload satellite imagery (triggers segmentation)
- `POST /ingest/document` — Upload PDF (OCR, chunk, embed, index)
- `POST /ingest/tweet` — Ingest tweet (sentiment + geocode)
- `POST /ingest/tweets/batch` — Batch tweet ingestion

### Jobs & Monitoring
- `GET /jobs/image/{job_id}` — Check image processing status
- `GET /jobs/image` — List recent image jobs
- `GET /dashboard/metrics` — Aggregate platform metrics

### Analysis
- `POST /rag/query` — Query knowledge base for expert commentary
- `POST /reports/generate` — Queue report generation (PDF/HTML/JSON)
- `GET /reports/{report_id}` — Check report status & download URL

### System
- `GET /health` — Health check
- `GET /` — API info

## Frontend Views

1. **Dashboard** — Metrics cards, damage pie chart, sentiment timeline, alerts, recent jobs
2. **Map Viewer** — Interactive Leaflet map with damage polygons & tweet sentiment heatmap
3. **Report Builder** — Generate reports + RAG expert query interface
4. **Data Ingest** — Upload imagery, documents, and tweets

## Kubernetes Deployment

```bash
# Build images and push to registry
docker build -t your-registry/pda-backend:latest backend/
docker build -t your-registry/pda-frontend:latest frontend/
docker push your-registry/pda-backend:latest
docker push your-registry/pda-frontend:latest

# Update image tags in infrastructure/k8s/backend.yaml and frontend.yaml
# Then deploy:
make k8s-deploy
```

## Testing

```bash
# Run backend tests
make test

# Or manually
cd backend && pytest tests/ -v
```

## Optional Full ML Dependencies

The backend Docker image installs a lean runtime by default so the API, PDF RAG,
mock CV inference, and lexicon sentiment fallback start reliably on normal
machines. Install full model-backed inference separately when you need local
Torch, Transformers, SentenceTransformers, Chroma, or Qdrant client support:

```bash
cd backend
python -m pip install -r requirements-ml.txt
```

## Project Structure

```
post-disaster-analytics/
├── backend/
│   ├── app/
│   │   ├── core/          # Config, DB, logging, storage, vector
│   │   ├── models/        # Pydantic schemas
│   │   ├── ml/            # Segmentation, sentiment, RAG engine
│   │   ├── pipelines/     # Imagery, document, social media ETL
│   │   ├── services/      # Report generator, GIS export
│   │   ├── api/           # FastAPI route modules
│   │   ├── db_models.py   # SQLAlchemy ORM
│   │   ├── init_db.py     # DB bootstrap
│   │   └── main.py        # FastAPI app entrypoint
│   ├── tests/             # Pytest suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # Dashboard, MapViewer, ReportBuilder, DataIngest
│   │   ├── services/      # Axios API client
│   │   ├── App.js
│   │   └── index.js
│   ├── public/
│   ├── Dockerfile
│   └── nginx.conf
├── infrastructure/
│   └── k8s/               # Kubernetes manifests
├── notebooks/
│   ├── 01_demo_pipeline.ipynb
│   └── 02_model_evaluation.ipynb
├── scripts/
│   └── seed_demo_data.py
├── docker-compose.yml
├── Makefile
└── README.md
```

## Technology Stack

| Layer            | Technologies                                          |
|------------------|-------------------------------------------------------|
| Web Framework    | FastAPI, React, Leaflet, Recharts                     |
| Database         | PostgreSQL + PostGIS, Qdrant, Redis                   |
| Storage          | MinIO (S3-compatible)                                 |
| ML / CV          | PyTorch, TorchVision, Transformers, SentenceTransformers |
| NLP / RAG        | LangChain, OpenAI, spaCy, rank-bm25                   |
| Geospatial       | Rasterio, GeoPandas, Shapely, GDAL                    |
| Reports          | ReportLab, Matplotlib, Jinja2                         |
| DevOps           | Docker, Kubernetes, Prometheus, Grafana                 |

## Configuration

Key environment variables (see `.env.example`):

- `DATABASE_URL` — PostgreSQL connection string
- `QDRANT_HOST` / `QDRANT_PORT` — Vector DB endpoint
- `REDIS_URL` — Redis connection
- `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` — Object storage
- `OPENAI_API_KEY` — Optional; enables GPT-4 RAG commentary (falls back to template mode)
- `JWT_SECRET` — Authentication signing key

## Notes

- **Mock Mode**: Without pre-trained weights or OpenAI key, the platform runs in deterministic mock/template mode suitable for demos and UI development.
- **GPU Support**: For production inference, add GPU nodes to K8s and update `torch` device handling in `app/ml/segmentation.py` and `app/ml/sentiment.py`.
- **Privacy**: All tweet user IDs are anonymized at ingestion. Ensure compliance with platform TOS and GDPR/CCPA.

## License

MIT
