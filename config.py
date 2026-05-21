# config.py — Shared constants for the RAG pipeline

# Books to ingest — each entry maps a display name to its PDF path inside books/
BOOKS = [
    {
        "book": "sapiens",
        "title": "Sapiens: A Brief History of Humankind",
        "path": "books/Sapiens A Brief History of Humankind.pdf",
    },
    {
        "book": "homo_deus",
        "title": "Homo Deus: A Brief History of Tomorrow",
        "path": "books/homo_deus_a_brief_history_of_tomorrow_pdf.pdf",
    },
    {
        "book": "21_lessons",
        "title": "21 Lessons for the 21st Century",
        "path": "books/21-lessons-for-the-21st-century-1.pdf",
    },
]

# Chroma vector database
COLLECTION_NAME = "harari_docs"
CHROMA_PERSIST_DIR = "./chroma_db"

# OpenAI models (used for embeddings; also used as LLM fallback if no Groq key)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Groq (free tier) — used as LLM when GROQ_API_KEY is set
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"

# Chunking strategy
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
RETRIEVAL_K = 3          # docs retrieved per query variant (multiquery base)
RERANK_TOP_N = 3         # docs kept after FlashRank reranking

# Relevance threshold for the similarity score returned by Chroma (L2 distance).
# Lower = more similar. Queries whose best chunk score exceeds this value
# are considered off-topic and are rejected before calling the LLM.
# Tune this value by checking the scores printed in the terminal.
RELEVANCE_THRESHOLD = 1.2
