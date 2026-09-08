import logging
import os
from pathlib import Path
from typing import Any, Dict, List
import pypdf
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# Arabic-friendly text separators prioritizing paragraphs, sentences, and punctuation
ARABIC_SEPARATORS = [
    "\n\n",
    "\n",
    "۔",  # Arabic full stop
    "؟",  # Arabic question mark
    "!",
    ".",
    "؛",  # Arabic semicolon
    "،",  # Arabic comma
    " ",
    "",
]


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initialize recursive character text splitter tailored for Arabic text."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=ARABIC_SEPARATORS,
        length_function=len,
    )


def extract_text_from_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF file page by page to maintain accurate page numbers.

    Returns:
        List of dicts: [{"text": page_text, "page": page_number}]
    """
    pages_data = []
    try:
        reader = pypdf.PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                cleaned = text.strip()
                if cleaned:
                    pages_data.append({"text": cleaned, "page": page_idx + 1})
            except Exception as page_err:
                logger.warning(f"Error extracting page {page_idx + 1} from {file_path.name}: {page_err}")
    except Exception as e:
        logger.error(f"Failed to read PDF {file_path}: {e}")
    return pages_data


def extract_text_from_docx(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from DOCX file.

    Returns:
        List of dicts: [{"text": doc_text, "page": 1}]
    """
    if DocxDocument is None:
        logger.error("python-docx is not installed.")
        return []

    try:
        doc = DocxDocument(str(file_path))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        joined = "\n\n".join(full_text)
        return [{"text": joined, "page": 1}] if joined else []
    except Exception as e:
        logger.error(f"Failed to read DOCX {file_path}: {e}")
        return []


def extract_text_from_txt(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from plain text file with encoding fallbacks for Arabic."""
    encodings = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read().strip()
                if content:
                    return [{"text": content, "page": 1}]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading TXT {file_path} with {enc}: {e}")
            break
    logger.error(f"Could not read {file_path} with any supported Arabic encoding.")
    return []


def load_and_chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    """Load a single document and split it into chunks with metadata.

    Returns:
        List of chunk dicts:
        [{
            "text": chunk_str,
            "metadata": {
                "source": file_name,
                "page": page_num,
                "chunk_id": f"{file_name}_p{page}_c{idx}"
            }
        }]
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        extracted = extract_text_from_pdf(file_path)
    elif suffix in [".docx", ".doc"]:
        extracted = extract_text_from_docx(file_path)
    elif suffix in [".txt", ".md"]:
        extracted = extract_text_from_txt(file_path)
    else:
        logger.info(f"Skipping unsupported file type: {file_path.name}")
        return []

    if not extracted:
        logger.warning(f"No text extracted from {file_path.name}")
        return []

    splitter = get_text_splitter()
    chunks: List[Dict[str, Any]] = []

    for page_item in extracted:
        page_text = page_item["text"]
        page_num = page_item["page"]

        page_chunks = splitter.split_text(page_text)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            cleaned_chunk = chunk_text.strip()
            if cleaned_chunk:
                chunks.append({
                    "text": cleaned_chunk,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_num,
                        "chunk_id": f"{file_path.stem}_p{page_num}_c{chunk_idx}",
                    },
                })

    return chunks


def get_all_data_files(data_dir: Path) -> List[Path]:
    """Scan data directory for supported documents."""
    supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
    if not data_dir.exists() or not data_dir.is_dir():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []

    files = [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]
    files.sort(key=lambda p: p.name)
    return files
