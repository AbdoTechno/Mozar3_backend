import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

_local_model: SentenceTransformer | None = None


def get_local_model() -> SentenceTransformer:
    """Load and cache the local lightweight multilingual model with offline-first speed."""
    global _local_model
    if _local_model is None:
        model_name = settings.LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading local multilingual embedding model ({model_name})...")
        try:
            # Try offline local files first for maximum speed (< 1s)
            _local_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            # Fallback to online download if not cached yet
            _local_model = SentenceTransformer(model_name)

        # Warm-up encoding so first query is instantaneous
        try:
            _local_model.encode(["warmup"], show_progress_bar=False)
        except Exception:
            pass

        logger.info("Local multilingual embedding model loaded & warmed up successfully.")
    return _local_model



def get_local_embedding(
    texts: Union[str, List[str]],
    batch_size: int = 32,
) -> List[List[float]]:
    """Generate dense vector embeddings locally without any API cost or rate limits.

    Args:
        texts: A string or list of strings in Arabic or English.
        batch_size: Batch size for CPU vectorization.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = get_local_model()
    single_input = isinstance(texts, str)
    items = [texts] if single_input else texts

    if not items:
        return []

    # Clean whitespace
    items_cleaned = [t.strip() if t.strip() else " " for t in items]

    embeddings = model.encode(
        items_cleaned,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return embeddings.tolist()
