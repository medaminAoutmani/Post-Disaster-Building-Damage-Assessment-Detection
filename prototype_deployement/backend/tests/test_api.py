import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_dashboard_metrics(client):
    r = await client.get("/dashboard/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_images_processed" in data
    assert "total_tweets_analyzed" in data

@pytest.mark.asyncio
async def test_rag_query(client):
    payload = {
        "query": "What are flood response priorities?",
        "damage_context": {"severity": "major"},
        "top_k": 3,
    }
    r = await client.post("/rag/query", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "commentary" in data
    assert "citations" in data
