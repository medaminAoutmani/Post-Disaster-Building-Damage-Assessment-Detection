import os
import io
import uuid
import hashlib
from typing import List, Dict, Any

from app.core.config import get_settings
from app.core.storage import upload_file
from app.core.vector import upsert_vectors
from app.core.db import AsyncSessionLocal
from app.db_models import Document, DocumentChunk
from app.ml.rag_engine import get_rag_engine
from app.core.logging import get_logger

logger = get_logger("pipelines.documents")
settings = get_settings()

class DocumentPipeline:
    def __init__(self):
        self.rag = get_rag_engine()

    def extract_text(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text and metadata from PDF bytes."""
        pages = []
        try:
            import fitz

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages.append({
                    "page": page_num + 1,
                    "text": text,
                    "word_count": len(text.split()),
                })
            doc.close()
        except Exception as e:
            logger.error("PDF extraction failed", error=str(e))
            raise
        return pages

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split pages into overlapping chunks."""
        full_text = "\n\n".join([p["text"] for p in pages])
        chunks = self._split_text(full_text, chunk_size=800, overlap=150)

        # Map chunks back to approximate page numbers
        result = []
        char_count = 0
        page_idx = 0
        for chunk in chunks:
            chunk_text = chunk
            # Rough page mapping
            while page_idx < len(pages) and char_count > sum(len(p["text"]) for p in pages[:page_idx+1]):
                page_idx += 1
            result.append({
                "text": chunk_text,
                "page": pages[min(page_idx, len(pages)-1)]["page"],
                "section": self._infer_section(chunk_text),
            })
            char_count += len(chunk_text)
        return result

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Small local splitter so document ingest does not require LangChain."""
        clean = " ".join(str(text or "").split())
        if not clean:
            return []

        chunks = []
        start = 0
        overlap = min(max(0, overlap), chunk_size // 2)
        while start < len(clean):
            end = min(len(clean), start + chunk_size)
            chunk = clean[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(clean):
                break
            start = end - overlap
        return chunks

    def _infer_section(self, text: str) -> str:
        """Heuristic section detection from chunk text."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["evacuation", "shelter", "relief"]):
            return "Response Actions"
        elif any(k in text_lower for k in ["assessment", "damage", "impact"]):
            return "Damage Assessment"
        elif any(k in text_lower for k in ["health", "medical", "sanitation"]):
            return "Health & Safety"
        return "General"

    async def ingest_document(self, source: str, title: str, doc_type: str, file_bytes: bytes, filename: str) -> str:
        """Full pipeline: extract, chunk, embed, index."""
        doc_uuid = uuid.uuid4()
        doc_id = str(doc_uuid)

        # Store raw file
        object_name = f"documents/{doc_id}/{filename}"
        upload_file(object_name, io.BytesIO(file_bytes), len(file_bytes), "application/pdf")

        # Extract and chunk
        pages = self.extract_text(file_bytes)
        chunks = self.chunk_document(pages)

        # Generate embeddings
        from app.ml.rag_engine import EmbeddingProvider
        embedder = EmbeddingProvider()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)

        # Store in DB and Vector DB
        async with AsyncSessionLocal() as session:
            doc = Document(
                id=doc_uuid,
                source=source,
                title=title,
                document_type=doc_type,
                file_url=object_name,
                total_pages=len(pages),
            )
            session.add(doc)

            vectors = []
            payloads = []
            db_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                vectors.append(embeddings[i].tolist())
                payloads.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "text": chunk["text"],
                    "source": source,
                    "title": title,
                    "section": chunk["section"],
                    "page": chunk["page"],
                })
                db_chunks.append(DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc_uuid,
                    text=chunk["text"],
                    section=chunk["section"],
                    page=chunk["page"],
                    vector_id=chunk_id,
                ))

            session.add_all(db_chunks)
            await session.commit()

        # Upsert to Qdrant (sync, but could be async)
        upsert_vectors(vectors, payloads, [p["chunk_id"] for p in payloads])

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        logger.info("Document indexed", doc_id=doc_id, title=title, chunks=len(chunks))
        return doc_id

    def _rebuild_bm25_index(self):
        """Rebuild sparse index from all chunks in DB."""
        # In production, query DB for all chunks; here we skip for simplicity
        # and rely on Qdrant dense search alone until BM25 is explicitly seeded.
        pass
