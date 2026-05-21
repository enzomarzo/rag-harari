---
title: Yuval Noah Harari RAG Assistant
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.13.0"
app_file: app.py
pinned: false
---

# Yuval Noah Harari: RAG Assistant

> **Course project** built during the *AI for Developers* course by [Code4All](https://codeforall.com),
> focused on building **RAG (Retrieval-Augmented Generation)** systems from scratch.

---

## Summary

Ask natural-language questions (in any language) about *Sapiens*, *Homo Deus*, and *21 Lessons for the 21st Century* and get answers grounded exclusively in the book content.

For a detailed explanation of the problem, the RAG technique, and every step of the pipeline, see [SUMMARY.md](SUMMARY.md).

---

## Table of Contents

1. [How it works (overview)](#1-how-it-works-overview)
2. [Project structure](#2-project-structure)
3. [Tech stack](#3-tech-stack)
4. [Setup & usage](#4-setup--usage)

---

## 1. How it works (overview)

```
books/
├── Sapiens: A Brief History of Humankind
├── Homo Deus: A Brief History of Tomorrow
└── 21 Lessons for the 21st Century

         ↓  ingestion.py  (run once)

   Chroma vector database  (local, ./chroma_db)
   — stores all book content as searchable embeddings

         ↓  inference.py  (runs on every question)

   1. Detect question language
   2. Translate to English if needed
   3. Classify which book the question is about
   4. Off-topic guard (fast similarity check)
   5. Multi-query retrieval + FlashRank reranking
   6. Generate answer with conversation memory
   7. Translate answer back if needed

         ↓  app.py

   Gradio chat interface (browser)
```

---

## 2. Project structure

```
rag-code4all/
│
├── books/                          ← PDF source files (not committed to git)
│   ├── Sapiens A Brief History of Humankind.pdf
│   ├── homo_deus_a_brief_history_of_tomorrow_pdf.pdf
│   └── 21-lessons-for-the-21st-century-1.pdf
│
├── chroma_db/                      ← local vector database (auto-created, not committed)
│
├── config.py                       ← all shared settings: book list, models, chunk sizes
├── ingestion.py                    ← Phase 1: load → clean → chunk → embed → store
├── inference.py                    ← Phase 2: detect → translate → classify → off-topic check → multi-query → rerank → answer
├── app.py                          ← Gradio chat UI
├── generate_index.py               ← optional: generates a chapter index for the UI
├── book_index.json                 ← output of generate_index.py
│
├── requirements.txt
├── .env.example                    ← template for environment variables
└── .gitignore
```

---

## 3. Tech stack

| Component | Library / Model | Purpose |
|---|---|---|
| Orchestration | `langchain` | Connects all components into chains |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text to vectors |
| LLM | OpenAI `gpt-4o-mini` | Generates and translates answers |
| Vector database | Chroma (local) | Stores and searches embeddings |
| Language detection | `langdetect` | Detects query language (no API call) |
| Multi-query retrieval | `MultiQueryRetriever` (LangChain) | Generates query variations for broader coverage |
| Reranking | `FlashRank` | Cross-encoder rescoring of candidate chunks |
| UI | Gradio | Browser-based chat interface |

---

## 4. Setup & usage

### Prerequisites

- Python 3.10+
- An OpenAI API key with available credits

### Step 1. Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2. Configure your API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3. Add the book PDFs

Place the three PDF files inside the `books/` folder. The expected filenames are defined in `config.py` under the `BOOKS` list.

### Step 4. Run ingestion (once)

```bash
python ingestion.py
```

This processes all three books and creates the `./chroma_db` folder. It only needs to run once. If a book is already stored, it is skipped automatically. To re-ingest everything from scratch, delete `./chroma_db` and run again.

### Step 5. Start the assistant

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser and start asking questions.


