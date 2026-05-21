# Project Deep Dive — Yuval Noah Harari RAG Assistant

> Detailed explanation of the problem, the RAG technique, and every step of the pipeline.
> For the quick-start guide see the [README](README.md).

---

## Table of Contents

1. [What problem does this project solve?](#1-what-problem-does-this-project-solve)
2. [What is RAG — and why does it matter?](#2-what-is-rag--and-why-does-it-matter)
3. [How the system works — step by step](#3-how-the-system-works--step-by-step)
   - [Phase 1 — Ingestion (one-time setup)](#phase-1--ingestion-one-time-setup)
   - [Phase 2 — Inference (every question)](#phase-2--inference-every-question)

---

## 1. What problem does this project solve?

Yuval Noah Harari wrote three books: **Sapiens**, **Homo Deus**, and **21 Lessons for the 21st Century**, that together span thousands of years of human history, predictions about the future, and reflections on the challenges of our time. Combined, they contain hundreds of pages of dense information.

Reading all three books to find a specific idea or answer a single question is impractical. Search tools like Ctrl+F only find exact words, not meaning. And if you ask ChatGPT directly, it answers from general training data, not from the actual text of these books.

**This project solves that:** it lets you ask any question in natural language (in any language), and the system finds the most relevant passages in the books and uses them to generate a precise, grounded answer.

---

## 2. What is RAG — and why does it matter?

**RAG** stands for **Retrieval-Augmented Generation**. It is a technique that combines two things:

- **Retrieval** => searching a knowledge base (in this case, the books) to find relevant text passages.
- **Generation** => using an AI language model (LLM) to formulate a clear, natural-language answer based on what was retrieved.

### Why not just ask ChatGPT directly?

When you ask a question to a general-purpose AI like ChatGPT:
- It answers from patterns learned during training and it may not have read these exact editions of the books.
- It can **hallucinate** (invent plausible-sounding but incorrect information).
- It cannot tell you *where* in the book it got the information.

With RAG:
- The AI can only use the text you gave it, which is the actual book content.
- If the answer is not in the books, the system says so instead of making something up.
- The answer is **grounded in the source material**.

### A simple analogy

Imagine you have a very smart assistant. Before answering your question, the assistant quickly skims through all three books, finds the 3 most relevant paragraphs, and then uses only those paragraphs to answer you. That is RAG.

---

## 3. How the system works — step by step

### Phase 1 — Ingestion (one-time setup)

This phase reads all the books, processes them, and stores the content in a way that makes intelligent search possible. You only run this once.

**Step 1 — Load the PDFs**
Each book is loaded page by page using `PyPDFLoader`. The result is a list of pages, each with its text content and page number.

**Step 2 — Clean the text**
Raw PDF text often contains formatting noise: multiple spaces, line breaks in the middle of sentences, stray page numbers. The text is cleaned with simple rules:
- All sequences of whitespace are collapsed into a single space.
- Trailing page numbers at the end of paragraphs are removed.

This improves the quality of what gets stored — and therefore the quality of future searches.

**Step 3 — Split into chunks**
An AI model cannot process a whole book at once. The text is split into small, overlapping **chunks** of ~1000 characters, with a 200-character overlap between consecutive chunks. The overlap ensures that sentences at the boundary between two chunks are not lost.

**Step 4 — Add metadata**
Each chunk is tagged with metadata:
- `book` — which book it came from (e.g. `"sapiens"`, `"homo_deus"`, `"21_lessons"`)
- `book_title` — the full title
- `page` — the page number in the original PDF (added automatically by the loader)

This metadata is used later during search to filter results to the right book.

**Step 5 — Create embeddings and store**
Each chunk is converted into an **embedding** — a list of numbers (a vector) that represents its *meaning*. Two chunks about similar topics will have vectors that are mathematically close to each other. These vectors, along with the original text and metadata, are stored in **Chroma**, a local vector database.

> **Why vectors?** Because searching by meaning is fundamentally different from searching by keyword. A keyword search for "agriculture" will not find a paragraph that talks about "farming" without using the word "agriculture". A vector search finds it — because the meaning is similar.

---

### Phase 2 — Inference (every question)

This phase runs every time a user asks a question.

**Step 1 — Detect language**
The question is analyzed statistically (no AI call, no cost) to detect what language it is written in — English, Portuguese, Spanish, etc. This is done using the `langdetect` library.

**Step 2 — Translate to English (if needed)**
All three books are in English, and the vector database stores English content. For the search to work correctly, the question must also be in English. If the user asked in Portuguese, the question is translated to English using the LLM before continuing.

**Step 3 — Classify which book the question is about**
A small LLM call (with `temperature=0` for determinism) reads the question and decides:
- Is this question specifically about *Sapiens*? → filter to `book = "sapiens"`
- Is it about *Homo Deus*? → filter to `book = "homo_deus"`
- Is it about *21 Lessons*? → filter to `book = "21_lessons"`
- Is it general or cross-book? → no filter, search all three

This is **metadata filtering** — it narrows the search space before any vector comparison happens, which makes retrieval faster and more accurate.

**Step 4 — Quick off-topic check**
A fast single-document similarity search checks whether the question is even related to any of the books. If the closest chunk in the database is still too far away (L2 distance above threshold), the system declines to answer rather than hallucinating. This guard runs before the heavier retrieval pipeline to avoid unnecessary cost.

**Step 5 — Multi-query retrieval**
Instead of searching with just one phrasing of the question, the LLM rewrites it in several different ways — more formal, more specific, from a different angle. Each variation is searched independently in Chroma, and all results are merged and deduplicated.

> **Why?** The same concept can be expressed in many ways. A question about "what changed when humans started farming" may not match a chunk that talks about "the Agricultural Revolution" with a simple embedding search. Multiple phrasings increase the chance of finding all relevant passages.

**Step 6 — FlashRank reranking**
After multi-query retrieval, there are many candidate chunks. A **cross-encoder** model (FlashRank) re-scores each chunk against the *original* question and keeps only the top 3 most relevant.

> **Cross-encoder vs embedding similarity:** Embedding similarity compares a query vector to document vectors independently — it is fast but approximate. A cross-encoder reads the query *and* the document together, giving a much more accurate relevance score. It is too slow to run on the entire database, but fast enough on the small candidate set from multi-query retrieval.

**Step 7 — Generate the answer with conversation memory**
The top reranked passages are assembled into a context block. The conversation history (last 5 turns) is also passed to the prompt, so the LLM can understand follow-up questions (e.g. "What did *he* say about that?"). The LLM is instructed to answer using only the provided documents.

**Step 8 — Translate back (if needed)**
If the original question was not in English, the answer is translated back to the user's language.

**Step 9 — Return with usage stats**
The answer is returned to the UI along with token counts and estimated cost, displayed as a footnote.
