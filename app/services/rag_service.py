import gc
import logging
from pathlib import Path
import time
from typing import List, Optional
from app.config import settings
from app.core.gemini_client import generate_answer, get_embedding
from app.core.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.schemas.rag_schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SourceItem,
)
from app.services.document_loader import get_all_data_files, load_and_chunk_document
from app.services.vector_store import (
    add_chunks_to_vector_store,
    get_indexed_files,
    mark_file_as_indexed,
    reset_vector_store,
    search_similar_chunks,
)

logger = logging.getLogger(__name__)


def ingest_documents_pipeline(force_reindex: bool = False) -> IngestResponse:
    """Execute ingestion pipeline for all supported agricultural files in the data directory.

    Args:
        force_reindex: If True, purges existing ChromaDB collection and starts fresh.

    Returns:
        IngestResponse with summary of indexed, skipped, and total chunks.
    """
    data_dir = settings.resolved_data_dir
    files = get_all_data_files(data_dir)

    if not files:
        return IngestResponse(
            status="warning",
            indexed_files=[],
            skipped_files=[],
            total_chunks_added=0,
            message=f"No supported documents (.pdf, .docx, .txt) found in {data_dir}",
        )

    if force_reindex:
        logger.info("Force re-indexing enabled: clearing existing ChromaDB collection...")
        reset_vector_store()
        already_indexed = set()
    else:
        already_indexed = get_indexed_files()

    indexed_files: List[str] = []
    skipped_files: List[str] = []
    total_chunks_added = 0

    logger.info(f"Found {len(files)} total files in {data_dir}. Already indexed: {len(already_indexed)}")

    for file_path in files:
        if file_path.name in already_indexed:
            logger.debug(f"Skipping already indexed file: {file_path.name}")
            skipped_files.append(file_path.name)
            continue

        logger.info(f"Processing and chunking document: {file_path.name}...")
        try:
            chunks = load_and_chunk_document(file_path)
            if not chunks:
                logger.warning(f"No valid text chunks generated for {file_path.name}")
                continue

            logger.info(
                f"Document '{file_path.name}' produced {len(chunks)} chunks. Generating embeddings..."
            )
            texts_to_embed = [c["text"] for c in chunks]
            embeddings = get_embedding(texts_to_embed, task_type="retrieval_document")

            chunks_added = add_chunks_to_vector_store(chunks, embeddings)
            total_chunks_added += chunks_added
            indexed_files.append(file_path.name)
            mark_file_as_indexed(file_path.name, chunks_added)
            logger.info(f"Successfully indexed and registered '{file_path.name}' ({chunks_added} chunks).")

            # Clean memory immediately to prevent OOM
            del chunks, texts_to_embed, embeddings
            gc.collect()

            # Gentle pacing for free-tier rate limits
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"Failed to ingest document '{file_path.name}': {e}", exc_info=True)
            gc.collect()
            time.sleep(5.0)

    summary_msg = (
        f"Ingestion complete: Indexed {len(indexed_files)} new files ({total_chunks_added} chunks), "
        f"skipped {len(skipped_files)} already-indexed files."
    )
    logger.info(summary_msg)

    return IngestResponse(
        status="success",
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        total_chunks_added=total_chunks_added,
        message=summary_msg,
    )


def process_farmer_chat(chat_request: ChatRequest) -> ChatResponse:
    """Execute complete RAG pipeline for an incoming farmer inquiry.

    1. Checks Gemini configuration.
    2. Embeds query.
    3. Retrieves top-K context chunks from ChromaDB.
    4. Assembles prompt with farmer profile and conversation history.
    5. Calls Gemini-1.5-flash with Egyptian agricultural expert persona.
    6. Returns structured response with cited sources.
    """
    if not settings.is_gemini_configured:
        return ChatResponse(
            answer=(
                "مرحباً بك في تطبيق المزارع! يبدو أن مفتاح الربط الخاص بـ Google Gemini API لم يتم تفعيله بعد "
                "في ملف الإعدادات (.env).\n"
                "يرجى وضع المفتاح الخاص بك في GEMINI_API_KEY لتفعيل المستشار الزراعي الذكي."
            ),
            sources=[],
            relevant_chunks_count=0,
        )

    # 1. Embed query
    query_embeddings = get_embedding(
        texts=chat_request.query,
        task_type="retrieval_query",
    )
    if not query_embeddings:
        raise ValueError("Failed to generate embedding for the farmer's query.")

    query_vec = query_embeddings[0]

    # 2. Similarity search in ChromaDB
    retrieved_chunks = search_similar_chunks(
        query_embedding=query_vec,
        top_k=settings.TOP_K_CHUNKS,
    )

    context_texts = [chunk["text"] for chunk in retrieved_chunks]

    # 3. Build Source citations
    sources: List[SourceItem] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        snippet = chunk["text"][:300].strip() + ("..." if len(chunk["text"]) > 300 else "")
        sources.append(
            SourceItem(
                document=meta.get("source", "نشرة إرشادية غير معروفة"),
                page=meta.get("page"),
                snippet=snippet,
            )
        )

    # 4. Synthesize prompt
    assembled_prompt = build_rag_prompt(
        query=chat_request.query,
        context_chunks=context_texts,
        farmer_context=chat_request.farmer_context,
        history=chat_request.history,
    )

    # 5. Generate response with Gemini
    answer_text = generate_answer(
        prompt=assembled_prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
    )

    return ChatResponse(
        answer=answer_text,
        sources=sources,
        relevant_chunks_count=len(retrieved_chunks),
    )
