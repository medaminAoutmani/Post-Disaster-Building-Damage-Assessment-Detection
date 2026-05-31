from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import uuid

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger("core.vector")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except Exception:  # pragma: no cover - exercised when qdrant-client is absent
    QdrantClient = None
    Distance = None
    PointStruct = None
    VectorParams = None


qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT) if QdrantClient else None
_memory_points: Dict[str, Dict[str, Any]] = {}
_collection_ready = False


def ensure_collection(vector_size: int = 384) -> bool:
    global _collection_ready
    if qdrant_client is None:
        return False
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == settings.QDRANT_COLLECTION for c in collections):
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        _collection_ready = True
        return True
    except Exception as exc:
        logger.warning("Qdrant unavailable; using in-memory vector fallback", error=str(exc))
        return False


def upsert_vectors(vectors: list, payloads: list, ids: Optional[list] = None):
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in vectors]

    for idx, point_id in enumerate(ids):
        _memory_points[str(point_id)] = {
            "id": str(point_id),
            "vector": vectors[idx],
            "payload": payloads[idx],
        }

    if not ensure_collection(vector_size=len(vectors[0]) if vectors else 384):
        return

    points = [
        PointStruct(id=str(ids[i]), vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]
    try:
        qdrant_client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    except Exception as exc:
        logger.warning("Qdrant upsert failed; kept vectors in memory", error=str(exc))


def search_vectors(query_vector: list, top_k: int = 5, filter_dict: Optional[dict] = None):
    if ensure_collection(vector_size=len(query_vector) if query_vector else 384):
        try:
            return qdrant_client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_dict,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed; using in-memory vector fallback", error=str(exc))

    scored = []
    for point in _memory_points.values():
        score = _cosine_similarity(query_vector, point["vector"])
        scored.append(SimpleNamespace(id=point["id"], score=score, payload=point["payload"]))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
