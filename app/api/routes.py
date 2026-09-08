import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.config import settings
from app.schemas.rag_schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    StatusResponse,
)
from app.services.document_loader import get_all_data_files
from app.services.rag_service import ingest_documents_pipeline, process_farmer_chat
from app.services.vector_store import get_vector_store_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Al-Mozare3 Agricultural RAG"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask an agricultural question with personalized farmer context",
    description="Processes farmer queries using ChromaDB retrieval and Gemini-1.5-flash with Egyptian expert guidelines.",
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat endpoint supporting farmer persona and document citations."""
    try:
        response = process_farmer_chat(request)
        return response
    except Exception as e:
        logger.error(f"Error handling chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء معالجة السؤال: {str(e)}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger document ingestion from data folder",
    description="Scans the data directory, splits documents, embeds chunks with Gemini, and stores vectors into ChromaDB.",
)
async def ingest_endpoint(request: IngestRequest = IngestRequest()) -> IngestResponse:
    """Manual trigger to index new documents or rebuild vector store."""
    if not settings.is_gemini_configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكن بدء الفهرسة لأن مفتاح GEMINI_API_KEY غير مهيأ في ملف .env",
        )

    try:
        response = ingest_documents_pipeline(force_reindex=request.force_reindex)
        return response
    except Exception as e:
        logger.error(f"Error during manual ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشلت عملية الفهرسة: {str(e)}",
        )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get backend and vector database statistics",
)
async def status_endpoint() -> StatusResponse:
    """Return current indexing status, count of files, and model configs."""
    stats = get_vector_store_stats()
    all_files = get_all_data_files(settings.resolved_data_dir)

    return StatusResponse(
        status="ready" if settings.is_gemini_configured else "needs_gemini_key",
        gemini_configured=settings.is_gemini_configured,
        total_documents_in_data=len(all_files),
        indexed_documents_count=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        embedding_model=settings.EMBEDDING_MODEL,
        chat_model=settings.CHAT_MODEL,
        indexed_files=stats.get("indexed_files", []),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def api_health() -> HealthResponse:
    """Return health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gemini_ready=settings.is_gemini_configured,
    )
