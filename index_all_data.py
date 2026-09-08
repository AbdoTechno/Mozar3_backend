"""Incremental Batch Ingestion Script for Al-Mozare3 RAG Backend.

Processes all documents in data/ one by one, with low memory usage (gc.collect()),
automatic rate-limit backoff, and immediate progress updates in indexed_registry.json.
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from app.config import settings
from app.core.gemini_client import get_embedding
from app.services.document_loader import get_all_data_files, load_and_chunk_document
from app.services.vector_store import (
    add_chunks_to_vector_store,
    get_indexed_files,
    mark_file_as_indexed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")


def index_all_documents():
    data_dir = settings.resolved_data_dir
    all_files = get_all_data_files(data_dir)

    if not all_files:
        logger.warning(f"No files found in {data_dir}")
        return

    # Sort files by size ascending (process fast, lightweight files first)
    all_files.sort(key=lambda p: p.stat().st_size)

    already_indexed = get_indexed_files()
    remaining = [f for f in all_files if f.name not in already_indexed]

    print("=" * 70)
    print(f"🌾 بدء فهرسة ملفات الداتا لمشروع المزارع (Al-Mozare3 Knowledge Base)")
    print(f"📁 إجمالي الملفات: {len(all_files)} | المفهرس حالياً: {len(already_indexed)} | المتبقي: {len(remaining)}")
    print("=" * 70)

    success_count = 0
    total_chunks_added = 0

    for idx, file_path in enumerate(remaining, 1):
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"\n[{idx}/{len(remaining)}] جاري معالجة: {file_path.name} ({file_size_mb:.2f} MB)...")

        try:
            # 1. Extract text and split
            chunks = load_and_chunk_document(file_path)
            if not chunks:
                print(f"  ⚠️ لم يتم استخراج نصوص كافية من: {file_path.name}")
                mark_file_as_indexed(file_path.name, 0)
                continue

            print(f"  📄 عدد المقاطع المستخرجة: {len(chunks)} مقطع. جاري التضمين...")

            # 2. Embed chunks in batches of 20
            texts = [c["text"] for c in chunks]
            embeddings = get_embedding(texts, task_type="retrieval_document", batch_size=20)

            # 3. Upsert into ChromaDB
            added = add_chunks_to_vector_store(chunks, embeddings, batch_size=100)
            total_chunks_added += added
            success_count += 1

            # 4. Record in registry immediately
            mark_file_as_indexed(file_path.name, added)
            print(f"  ✅ تمت الفهرسة بنجاح وحفظ {added} متجه في قاعدة البيانات.")

            # 5. Free memory immediately
            del chunks, texts, embeddings
            gc.collect()

            # Gentle pause between files to respect rate limits
            time.sleep(1.5)

        except Exception as e:
            logger.error(f"  ❌ خطأ أثناء فهرسة {file_path.name}: {e}")
            gc.collect()
            time.sleep(5.0)

    print("\n" + "=" * 70)
    print(f"🎉 اكتملت الفهرسة: تم إضافة {success_count} ملف جديد ({total_chunks_added} متجه إضافي).")
    print(f"📊 إجمالي الملفات المسجلة الآن: {len(get_indexed_files())} ملف.")
    print("=" * 70)


if __name__ == "__main__":
    index_all_documents()
