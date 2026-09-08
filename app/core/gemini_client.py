import logging
import time
from typing import List, Optional, Union
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

_is_configured = False


def ensure_gemini_configured() -> bool:
    """Initialize and configure Google Generative AI SDK if not already configured."""
    global _is_configured
    if _is_configured:
        return True

    if not settings.is_gemini_configured:
        logger.warning(
            "GEMINI_API_KEY is not configured or uses placeholder! Please set it in .env file."
        )
        return False

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY, transport="rest")
        _is_configured = True
        logger.info("Google Gemini SDK successfully configured (REST transport).")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini SDK: {e}")
        return False


def get_embedding(
    texts: Union[str, List[str]],
    task_type: str = "retrieval_document",
    batch_size: int = 32,
) -> List[List[float]]:
    """Compute vector embeddings using local sentence-transformers or Gemini.

    Args:
        texts: A single text string or list of text strings.
        task_type: 'retrieval_document' or 'retrieval_query'.
        batch_size: Max number of items per batch.

    Returns:
        List of embedding vectors (list of floats).
    """
    if settings.EMBEDDING_PROVIDER == "local":
        from app.core.local_embedding import get_local_embedding
        return get_local_embedding(texts, batch_size=batch_size)

    if not ensure_gemini_configured():
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
        )

    single_input = isinstance(texts, str)
    items = [texts] if single_input else texts

    if not items:
        return []

    results: List[List[float]] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        # Clean empty texts
        batch_cleaned = [t.strip() if t.strip() else " " for t in batch]

        max_retries = 5
        backoff = 5.0
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                response = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=batch_cleaned,
                    task_type=task_type,
                )
                embeddings = response.get("embedding", [])
                if embeddings and isinstance(embeddings[0], float):
                    embeddings = [embeddings]

                results.extend(embeddings)
                success = True
                break
            except Exception as exc:
                err_str = str(exc)
                is_quota = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                wait_time = 30.0 if is_quota else backoff

                logger.warning(
                    f"Embedding batch [{i}:{i+len(batch)}] attempt {attempt}/{max_retries} encountered "
                    f"{'Rate Limit/Quota (429)' if is_quota else 'Error'}: {exc}. Waiting {wait_time}s..."
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    backoff *= 2
                else:
                    logger.error(f"Embedding failed completely for batch [{i}:{i+len(batch)}]: {exc}")
                    raise exc

    return results


def generate_answer(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Generate agricultural answer using Gemini LLM.

    Args:
        prompt: Assembled context & query prompt.
        system_instruction: System guidelines and persona instructions.
        temperature: Sampling temperature (low for factual adherence).

    Returns:
        Generated text answer.
    """
    if not ensure_gemini_configured():
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
        )

    try:
        model = genai.GenerativeModel(
            model_name=settings.CHAT_MODEL,
            system_instruction=system_instruction,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                max_output_tokens=2048,
            ),
        )

        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "عذراً، لم أتمكن من تكوين إجابة كاملة من المصادر المتوفرة حالياً."
    except Exception as e:
        logger.error(f"Error during Gemini generation: {e}")
        raise e
