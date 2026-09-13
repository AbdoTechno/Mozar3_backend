import asyncio
from contextlib import asynccontextmanager
import logging
import sys
from typing import AsyncGenerator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.config import settings
from app.schemas.rag_schemas import HealthResponse
from app.services.document_loader import get_all_data_files
from app.services.rag_service import ingest_documents_pipeline
from app.services.vector_store import get_vector_store_stats

# Configure clean logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("al_mozare3")


def startup_ingestion_task() -> None:
    """Run ingestion check on startup."""
    stats = get_vector_store_stats()
    all_files = get_all_data_files(settings.resolved_data_dir)

    logger.info(
        f"Vector DB contains {stats.get('total_chunks', 0)} chunks across "
        f"{stats.get('total_documents', 0)} indexed files."
    )
    logger.info(f"Data directory has {len(all_files)} total supported files.")

    if not settings.is_gemini_configured:
        logger.warning(
            "⚠️ GEMINI_API_KEY is not configured in .env! "
            "Automated startup ingestion skipped. Add your key and restart, or call POST /api/ingest."
        )
        return

    if settings.AUTO_INGEST_ON_STARTUP:
        indexed_count = stats.get("total_documents", 0)
        total_files = len(all_files)

        if indexed_count < total_files:
            logger.info(
                f"Starting background incremental ingestion ({total_files - indexed_count} unindexed files)..."
            )
            try:
                ingest_documents_pipeline(force_reindex=False)
            except Exception as e:
                logger.error(f"Startup ingestion encountered an error: {e}", exc_info=True)
        else:
            logger.info("All documents are already indexed in ChromaDB. Ready to serve requests!")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown."""
    logger.info("==================================================================")
    logger.info("🌱 Starting Al-Mozare3 RAG Backend (المزارع) - Smart Agriculture 🌱")
    logger.info(f"📍 Storage Directory: {settings.resolved_chroma_dir}")
    logger.info(f"📁 Documents Directory: {settings.resolved_data_dir}")
    logger.info(f"🤖 Chat Model: {settings.CHAT_MODEL} | Embedding: {settings.EMBEDDING_MODEL}")
    logger.info("==================================================================")

    # Pre-initialize Chroma collection synchronously to ensure schema is initialized
    from app.services.vector_store import get_collection
    try:
        get_collection()
    except Exception as e:
        logger.warning(f"Chroma collection initialization warning: {e}")

    # Warm up local embedding model in memory for instant queries
    if settings.EMBEDDING_PROVIDER == "local":
        from app.core.local_embedding import get_local_model
        try:
            get_local_model()
        except Exception as e:
            logger.warning(f"Local embedding warm-up warning: {e}")

    # Run ingestion check asynchronously in background thread so server starts immediately
    asyncio.get_running_loop().run_in_executor(None, startup_ingestion_task)

    yield

    logger.info("Shutting down Al-Mozare3 Backend.")



app = FastAPI(
    title="Al-Mozare3 RAG Backend (المزارع)",
    description=(
        "Production-grade, local-first RAG backend for the 'Al-Mozare3' smart agriculture application. "
        "Powered by Google Gemini 1.5 Flash, Gemini text-embedding-004, and persistent ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration - Allow all origins for Flutter mobile app, emulators, and local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/", tags=["General"])
async def root():
    """Root landing endpoint with system summary and links."""
    return {
        "app": "Al-Mozare3 RAG Backend (المزارع)",
        "version": "1.0.0",
        "description": "Smart Egyptian Agricultural Advisor RAG API",
        "documentation": "/docs",
        "status_endpoint": "/api/status",
        "chat_endpoint": "/api/chat",
        "health_endpoint": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon dummy route to prevent 404 in web browsers."""
    from fastapi import Response
    return Response(status_code=204)


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Top-level health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gemini_ready=settings.is_gemini_configured,
    )



if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

