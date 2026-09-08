import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import chromadb
from app.config import settings

logger = logging.getLogger(__name__)

def get_collection_name() -> str:
    """Get collection name based on embedding provider to ensure dimension consistency."""
    return f"al_mozare3_{settings.EMBEDDING_PROVIDER}_knowledge"

def get_registry_filename() -> str:
    """Get registry filename based on embedding provider."""
    return f"indexed_registry_{settings.EMBEDDING_PROVIDER}.json"

_client: Optional[chromadb.ClientAPI] = None
_collection = None


def get_registry_path() -> Path:
    """Get path to the JSON registry file storing indexed documents metadata."""
    storage_path = settings.resolved_chroma_dir
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path / get_registry_filename()


def load_registry() -> Dict[str, Any]:
    """Load indexed documents registry from local JSON file."""
    reg_path = get_registry_path()
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading indexed registry: {e}. Rebuilding...")

    # If registry doesn't exist on disk yet, inspect ChromaDB once and persist
    registry = {}
    try:
        col = get_collection()
        if col.count() > 0:
            results = col.get(include=["metadatas"])
            for meta in results.get("metadatas", []):
                if meta and "source" in meta:
                    src = meta["source"]
                    registry[src] = registry.get(src, 0) + 1
            save_registry(registry)
    except Exception as e:
        logger.warning(f"Could not build registry from Chroma: {e}")

    return registry


def save_registry(registry: Dict[str, Any]) -> None:
    """Save registry to local JSON file."""
    try:
        reg_path = get_registry_path()
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save registry to {reg_path}: {e}")


def mark_file_as_indexed(filename: str, chunk_count: int) -> None:
    """Record a newly indexed file in the lightweight registry."""
    registry = load_registry()
    registry[filename] = {
        "chunk_count": chunk_count,
        "indexed_at": str(Path(__file__).stat().st_mtime),
    }
    save_registry(registry)


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or initialize persistent ChromaDB client."""
    global _client
    if _client is None:
        storage_path = settings.resolved_chroma_dir
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB persistent storage at: {storage_path}")
        _client = chromadb.PersistentClient(path=str(storage_path))
    return _client


def get_collection():
    """Get or create the knowledge collection in ChromaDB."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        col_name = get_collection_name()
        _collection = client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_indexed_files() -> Set[str]:
    """Retrieve set of unique filenames currently indexed using the fast registry."""
    registry = load_registry()
    return set(registry.keys())


def add_chunks_to_vector_store(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = 250,
) -> int:
    """Insert document chunks and their precomputed embeddings into ChromaDB in batches.

    Returns:
        Total number of chunks successfully inserted.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
        )

    col = get_collection()
    total = len(chunks)
    added = 0

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_chunks = chunks[i:end]
        batch_embeddings = embeddings[i:end]

        ids = [
            chunk["metadata"].get("chunk_id", f"chunk_{i + idx}")
            for idx, chunk in enumerate(batch_chunks)
        ]
        documents = [chunk["text"] for chunk in batch_chunks]
        metadatas = [chunk["metadata"] for chunk in batch_chunks]

        col.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        added += len(batch_chunks)
        logger.info(f"Upserted vector batch [{i}:{end}] of {total} chunks.")

    return added


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Perform vector similarity search against Chroma collection."""
    col = get_collection()
    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "text": doc,
            "metadata": meta or {},
            "distance": float(dist),
        })

    return output


def get_vector_store_stats() -> Dict[str, Any]:
    """Get summary statistics about the ChromaDB collection."""
    try:
        col = get_collection()
        total_chunks = col.count()
        indexed_files = list(get_indexed_files())
        indexed_files.sort()
        return {
            "total_chunks": total_chunks,
            "total_documents": len(indexed_files),
            "indexed_files": indexed_files,
        }
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "indexed_files": [],
        }


def reset_vector_store() -> None:
    """Clear and recreate the collection and registry."""
    global _collection
    client = get_chroma_client()
    col_name = get_collection_name()
    try:
        client.delete_collection(name=col_name)
        logger.info(f"Deleted Chroma collection: {col_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection (may not exist): {e}")

    _collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    # Clear registry file
    save_registry({})
    logger.info(f"Re-created empty collection and cleared registry: {col_name}")
