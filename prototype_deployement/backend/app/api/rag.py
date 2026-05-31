from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQueryRequest, RAGCommentary
from app.ml.rag_engine import get_rag_engine
from app.core.logging import get_logger

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = get_logger("api.rag")

@router.get("/status")
async def rag_status():
    """Show local PDF corpus and indexing status."""
    engine = get_rag_engine()
    return engine.local_pdf_status()

@router.post("/query", response_model=RAGCommentary)
async def rag_query(req: RAGQueryRequest):
    """Query the RAG engine for expert commentary."""
    try:
        engine = get_rag_engine()
        result = await engine.generate_commentary(
            query=req.query,
            damage_context=req.damage_context,
            top_k=req.top_k,
        )
        return RAGCommentary(
            commentary=result["commentary"],
            citations=result["citations"],
            confidence=result["confidence"],
            retrieved_chunks=[
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c.get("document_id", "unknown"),
                    "text": c["text"],
                    "source": c.get("source", "Unknown source"),
                    "section": c.get("section"),
                    "page": c.get("page"),
                    "score": c.get("score"),
                }
                for c in result["retrieved_chunks"]
            ],
        )
    except Exception as e:
        logger.error("RAG query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
