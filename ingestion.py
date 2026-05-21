"""
ingestion.py — Ingestion pipeline

Loads all books defined in config.BOOKS, splits them into chunks, adds book
metadata to each chunk, and stores embeddings in a local Chroma collection.

Run once before using the app:
    python ingestion.py

To re-ingest from scratch, delete the ./chroma_db folder and run again.
"""

import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    BOOKS,
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()


def _clean_text(text: str) -> str:
    """Remove PDF artefacts: multiple whitespace and trailing page numbers."""
    text = re.sub(r'\s+', ' ', text)                   # collapse whitespace / newlines
    text = re.sub(r'(?<=\.)\s*\d+\s*$', '', text)      # remove trailing page numbers
    return text.strip()


def ingest_documents():
    print("-" * 80)
    print("INGESTION PIPELINE")
    print("-" * 80)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    chunk_id = 0  # global counter used as unique Chroma document ID

    for book_cfg in BOOKS:
        book_key = book_cfg["book"]
        book_title = book_cfg["title"]
        pdf_path = book_cfg["path"]

        print(f"\n{'=' * 80}")
        print(f"Book: {book_title}")
        print(f"{'=' * 80}")

        # Guard: skip if this book is already stored
        existing = vectorstore._collection.get(where={"book": book_key})
        if existing and existing["ids"]:
            print(f"  ✓ Already ingested ({len(existing['ids'])} chunks). Skipping.")
            chunk_id += len(existing["ids"])
            continue

        # -------------------------------------------------------------------------
        # STEP 1: LOAD
        # -------------------------------------------------------------------------
        print(f"\n  [1/4] Loading PDF...")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF not found at '{pdf_path}'. "
                "Make sure the file is inside the books/ folder."
            )
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"  ✓ Loaded {len(documents)} pages")

        # -------------------------------------------------------------------------
        # STEP 2: CLEAN TEXT
        # -------------------------------------------------------------------------
        print(f"\n  [2/4] Cleaning text...")
        for doc in documents:
            doc.page_content = _clean_text(doc.page_content)
        print(f"  ✓ Cleaned {len(documents)} pages")

        # -------------------------------------------------------------------------
        # STEP 3: CHUNK
        # -------------------------------------------------------------------------
        print(f"\n  [3/4] Chunking...")
        chunks = splitter.split_documents(documents)
        print(f"  ✓ Split into {len(chunks)} chunks")

        # -------------------------------------------------------------------------
        # STEP 4: ADD METADATA + STORE
        # -------------------------------------------------------------------------
        print(f"\n  [4/4] Adding metadata and storing in Chroma...")
        for chunk in chunks:
            chunk.metadata["book"] = book_key          # e.g. "sapiens"
            chunk.metadata["book_title"] = book_title  # e.g. "Sapiens: A Brief History..."

        ids = [str(chunk_id + i) for i in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=ids)
        chunk_id += len(chunks)
        total_chunks += len(chunks)
        print(f"  ✓ Stored {len(chunks)} chunks (metadata: book='{book_key}')")

    print(f"\n{'-' * 80}")
    print(f"INGESTION COMPLETE — {total_chunks} new chunks stored across {len(BOOKS)} books")
    print("Run 'python app.py' to start the assistant")
    print("-" * 80)


if __name__ == "__main__":
    ingest_documents()
