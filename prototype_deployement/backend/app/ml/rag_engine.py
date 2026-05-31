import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.vector import search_vectors, upsert_vectors

settings = get_settings()
logger = get_logger("ml.rag")


DEFAULT_KNOWLEDGE_CHUNKS = [
    {
        "chunk_id": "default_flood_response",
        "document_id": "built_in",
        "text": (
            "For flood response, prioritize life safety, rapid needs assessment, evacuation support, "
            "temporary shelter, safe drinking water, sanitation, and restoration of critical access routes."
        ),
        "source": "Built-in disaster response baseline",
        "title": "Flood Response Baseline",
        "section": "Response Actions",
        "page": None,
    },
    {
        "chunk_id": "default_damage_assessment",
        "document_id": "built_in",
        "text": (
            "Major or destroyed building labels should trigger field validation, search-and-rescue triage, "
            "engineering inspection, debris clearance planning, and confidence review before irreversible decisions."
        ),
        "source": "Built-in damage assessment baseline",
        "title": "Damage Assessment Baseline",
        "section": "Damage Assessment",
        "page": None,
    },
    {
        "chunk_id": "default_public_sentiment",
        "document_id": "built_in",
        "text": (
            "High fear, sadness, or negative sentiment in public posts can indicate unmet needs, information gaps, "
            "and psychosocial stress. Aggregate analysis should avoid individual profiling."
        ),
        "source": "Built-in social impact baseline",
        "title": "Social Impact Baseline",
        "section": "Public Sentiment",
        "page": None,
    },
]


class EmbeddingProvider:
    """Embedding facade with online, local, and deterministic fallback modes."""

    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
        self.mode = "hash"
        self.model = None
        self._load_model()

    def _load_model(self):
        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import OpenAIEmbeddings

                self.model = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                )
                self.mode = "openai"
                logger.info("Using OpenAI embeddings", model=settings.EMBEDDING_MODEL)
                return
            except Exception as exc:
                logger.warning("OpenAI embeddings unavailable; trying local embeddings", error=str(exc))

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                settings.LOCAL_EMBEDDING_MODEL,
                local_files_only=not settings.ALLOW_REMOTE_MODEL_DOWNLOADS,
            )
            self.mode = "sentence_transformers"
            logger.info("Using local sentence-transformer embeddings", model=settings.LOCAL_EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Using deterministic hash embeddings", error=str(exc))

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = [str(text or "") for text in texts]
        if not texts:
            return np.empty((0, self.vector_size), dtype="float32")

        if self.mode == "openai" and self.model is not None:
            vectors = self.model.embed_documents(texts)
            return np.asarray(vectors, dtype="float32")

        if self.mode == "sentence_transformers" and self.model is not None:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype="float32")

        return np.asarray([self._hash_embed(text) for text in texts], dtype="float32")

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0].tolist()

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.vector_size, dtype="float32")
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm


class RAGEngine:
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.indexed_pdf_chunks = 0
        if settings.RAG_AUTO_INDEX_LOCAL_PDFS:
            self.indexed_pdf_chunks = self.index_local_pdf_folder()

    async def generate_commentary(
        self,
        query: str,
        damage_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        query_text = self._build_query_text(query, damage_context)
        query_vector = self.embedder.embed_query(query_text)
        retrieved = self._retrieve(query_text, query_vector, max(1, min(top_k, 20)))
        commentary = self._template_commentary(query, damage_context or {}, retrieved)
        citations = [self._citation(chunk) for chunk in retrieved]
        confidence = self._confidence(retrieved)

        return {
            "commentary": commentary,
            "citations": citations,
            "confidence": confidence,
            "retrieved_chunks": retrieved,
        }

    def index_local_pdf_folder(self, pdf_dir: Optional[str] = None) -> int:
        folder = self._resolve_pdf_dir(pdf_dir or settings.RAG_PDF_DIR)
        if folder is None:
            logger.info("RAG PDF folder not found; using built-in fallback context", path=pdf_dir or settings.RAG_PDF_DIR)
            return 0

        pdf_paths = sorted(folder.glob("*.pdf"))
        if not pdf_paths:
            logger.info("RAG PDF folder is empty; using built-in fallback context", path=str(folder))
            return 0

        chunks = []
        for pdf_path in pdf_paths:
            try:
                chunks.extend(self._chunks_from_pdf(pdf_path))
            except Exception as exc:
                logger.warning("Failed to index RAG PDF", path=str(pdf_path), error=str(exc))

        if not chunks:
            logger.warning("No text chunks extracted from RAG PDFs", path=str(folder))
            return 0

        vectors = self.embedder.embed([chunk["text"] for chunk in chunks])
        ids = [chunk["chunk_id"] for chunk in chunks]
        upsert_vectors([vector.tolist() for vector in vectors], chunks, ids)
        logger.info("Indexed local RAG PDFs", path=str(folder), pdfs=len(pdf_paths), chunks=len(chunks))
        return len(chunks)

    def local_pdf_status(self) -> Dict[str, Any]:
        folder = self._resolve_pdf_dir(settings.RAG_PDF_DIR)
        pdfs = []
        if folder is not None:
            pdfs = [
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(folder.glob("*.pdf"))
            ]
        return {
            "pdf_dir": str(folder) if folder else settings.RAG_PDF_DIR,
            "pdf_count": len(pdfs),
            "indexed_pdf_chunks": self.indexed_pdf_chunks,
            "embedding_mode": self.embedder.mode,
            "pdfs": pdfs,
        }

    def _resolve_pdf_dir(self, raw_path: str) -> Optional[Path]:
        path = Path(raw_path)
        repo_root = Path(__file__).resolve().parents[3]
        backend_root = Path(__file__).resolve().parents[2]
        candidates = [
            path,
            Path.cwd() / path,
            backend_root / path,
            repo_root / path,
            repo_root / "pdf_rag",
            Path("/app/pdf_rag"),
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved
        return None

    def _chunks_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        pages = self._extract_pdf_pages(pdf_path)
        chunks = []
        for page_number, text in pages:
            for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                chunk_id = self._stable_chunk_id(pdf_path, page_number, chunk_index, chunk_text)
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": self._stable_document_id(pdf_path),
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "title": pdf_path.stem.replace("_", " "),
                    "section": self._infer_section(chunk_text),
                    "page": page_number,
                })
        return chunks

    def _extract_pdf_pages(self, pdf_path: Path) -> List[tuple]:
        try:
            import fitz

            pages = []
            with fitz.open(pdf_path) as doc:
                for page_index, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append((page_index, text))
            return pages
        except Exception as fitz_exc:
            try:
                import pdfplumber

                pages = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page_index, page in enumerate(pdf.pages, start=1):
                        text = (page.extract_text() or "").strip()
                        if text:
                            pages.append((page_index, text))
                return pages
            except Exception as plumber_exc:
                raise RuntimeError(f"PyMuPDF failed: {fitz_exc}; pdfplumber failed: {plumber_exc}") from plumber_exc

    def _chunk_text(self, text: str) -> List[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return []

        chunk_size = max(300, settings.RAG_CHUNK_SIZE)
        overlap = min(max(0, settings.RAG_CHUNK_OVERLAP), chunk_size // 2)
        chunks = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + chunk_size)
            chunk = clean[start:end].strip()
            if len(chunk) > 80:
                chunks.append(chunk)
            if end == len(clean):
                break
            start = end - overlap
        return chunks

    def _stable_document_id(self, pdf_path: Path) -> str:
        digest = hashlib.sha1(str(pdf_path.name).encode("utf-8")).hexdigest()
        return f"pdf_{digest[:16]}"

    def _stable_chunk_id(self, pdf_path: Path, page_number: int, chunk_index: int, text: str) -> str:
        raw = f"{pdf_path.name}:{page_number}:{chunk_index}:{text[:120]}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"pdf_{digest}"

    def _infer_section(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ("evacuation", "shelter", "relief", "response", "recovery")):
            return "Response Actions"
        if any(term in lowered for term in ("damage", "impact", "assessment", "loss", "risk")):
            return "Damage and Impact"
        if any(term in lowered for term in ("health", "sanitation", "water", "disease", "medical")):
            return "Health and Safety"
        if any(term in lowered for term in ("economic", "poverty", "finance", "income", "cost")):
            return "Economic Impact"
        return "General"

    def _build_query_text(self, query: str, damage_context: Optional[Dict[str, Any]]) -> str:
        context_bits = []
        for key, value in (damage_context or {}).items():
            context_bits.append(f"{key}: {value}")
        return " ".join([query, *context_bits]).strip()

    def _retrieve(self, query_text: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        hits = search_vectors(query_vector, top_k=top_k)
        chunks = [self._normalise_hit(hit) for hit in hits if getattr(hit, "payload", None)]
        if not chunks:
            chunks = DEFAULT_KNOWLEDGE_CHUNKS[:top_k]

        query_terms = set(re.findall(r"[a-z0-9']+", query_text.lower()))
        for chunk in chunks:
            chunk["keyword_score"] = self._keyword_score(query_terms, chunk.get("text", ""))
        return sorted(
            chunks,
            key=lambda item: (float(item.get("score", 0.0)) * 0.7) + (item.get("keyword_score", 0.0) * 0.3),
            reverse=True,
        )[:top_k]

    def _normalise_hit(self, hit: Any) -> Dict[str, Any]:
        payload = dict(getattr(hit, "payload", {}) or {})
        chunk_id = payload.get("chunk_id") or str(getattr(hit, "id", "unknown"))
        return {
            "chunk_id": chunk_id,
            "document_id": payload.get("document_id", "unknown"),
            "text": payload.get("text", ""),
            "source": payload.get("source", "Unknown source"),
            "title": payload.get("title"),
            "section": payload.get("section"),
            "page": payload.get("page"),
            "score": round(float(getattr(hit, "score", 0.0) or 0.0), 4),
        }

    def _keyword_score(self, query_terms: set, text: str) -> float:
        text_terms = set(re.findall(r"[a-z0-9']+", text.lower()))
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / math.sqrt(len(query_terms) * len(text_terms))

    def _template_commentary(
        self,
        query: str,
        damage_context: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> str:
        severity = str(damage_context.get("severity", "")).lower()
        priorities = [
            "validate model outputs with field reports before high-impact operational decisions",
            "triage zones with major or destroyed labels for search, rescue, and engineering inspection",
            "coordinate shelter, water, sanitation, and medical support for affected communities",
        ]
        if "flood" in query.lower():
            priorities.insert(1, "confirm evacuation routes, water extent, and safe access corridors")
        if severity in {"major", "destroyed", "severe"}:
            priorities.insert(0, "treat the mapped area as high priority until ground truth confirms otherwise")

        evidence = "; ".join(
            f"{chunk.get('source', 'Unknown source')} / {chunk.get('section') or 'General'}"
            for chunk in chunks[:3]
        )
        return (
            "Based on the retrieved disaster-response context, responders should "
            + "; ".join(priorities[:4])
            + f". Evidence used: {evidence}."
        )

    def _citation(self, chunk: Dict[str, Any]) -> str:
        source = chunk.get("source") or chunk.get("title") or "Unknown source"
        section = chunk.get("section")
        page = chunk.get("page")
        parts = [source]
        if section:
            parts.append(str(section))
        if page:
            parts.append(f"p. {page}")
        return " - ".join(parts)

    def _confidence(self, chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.25
        scores = [max(0.0, min(1.0, float(chunk.get("score", 0.0) or 0.0))) for chunk in chunks]
        if not any(scores):
            return 0.55
        return round(min(0.95, 0.45 + (sum(scores) / len(scores)) * 0.5), 3)


_rag_engine = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
