"""
inference.py — Inference pipeline (RAG chain)

Performs a similarity search in Chroma, formats the retrieved context,
and sends it to the LLM via a structured prompt to generate an answer.

Can be imported by app.py or called directly:
    python inference.py
"""

import os
from dotenv import load_dotenv
from langdetect import detect as detect_lang
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq

# Maps ISO 639-1 codes (from langdetect) to full language names (for the translation prompt)
LANGUAGE_NAMES = {
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}

from config import (
    BOOKS,
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    GROQ_LLM_MODEL,
    RETRIEVAL_K,
    RERANK_TOP_N,
    RELEVANCE_THRESHOLD,
)

# Map book key → display title (built from config at import time)
BOOK_TITLES = {b["book"]: b["title"] for b in BOOKS}
BOOK_KEYS = [b["book"] for b in BOOKS]  # e.g. ["sapiens", "homo_deus", "21_lessons"]

load_dotenv()

# ---------------------------------------------------------------------------
# Shared resources (initialised once at import time)
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# LLM: use Groq (free) if GROQ_API_KEY is set, otherwise fallback to OpenAI
USE_GROQ = bool(os.getenv("GROQ_API_KEY"))
if USE_GROQ:
    llm = ChatGroq(model=GROQ_LLM_MODEL)
    print(f"[LLM] Using Groq ({GROQ_LLM_MODEL})")
else:
    llm = ChatOpenAI(model=LLM_MODEL)
    print(f"[LLM] Using OpenAI ({LLM_MODEL})")

vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Translates text to English
TRANSLATE_TO_EN_TEMPLATE = ChatPromptTemplate.from_template(
    "Translate the following text to English. "
    "Reply with only the translated text, nothing else.\n\nText: {text}"
)

# Translates text to a target language
TRANSLATE_TO_LANG_TEMPLATE = ChatPromptTemplate.from_template(
    "Translate the following text to {language}. "
    "Reply with only the translated text, nothing else.\n\nText: {text}"
)

# Classifies query into one of the known book keys, or "all"
CLASSIFY_BOOK_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a routing assistant. Decide which book the question is specifically about.

Available books:
- sapiens         → "Sapiens: A Brief History of Humankind"
- homo_deus       → "Homo Deus: A Brief History of Tomorrow"
- 21_lessons      → "21 Lessons for the 21st Century"

If the question is clearly about ONE book, reply with only its key (e.g. sapiens).
If the question spans multiple books or is general, reply with: all

Question: {query}

Book key:"""
)

# RAG answer — receives English query, English context, and formatted history
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant answering questions about Yuval Noah Harari's books.

    Use the conversation history (if any) and the provided documents to answer the question.
    If the question references something from the history (e.g. "it", "that", "the author"),
    use the history to understand what is being referred to.
    If the answer is not in the documents, say so. Do not use any other information.

    Conversation History:
    {history}

    Documents:
    {context}

    Question: {query}

    Answer:
    """
)

# Chains (each: prompt → LLM → plain string)
_llm_zero = ChatGroq(model=GROQ_LLM_MODEL, temperature=0) if USE_GROQ else ChatOpenAI(model=LLM_MODEL, temperature=0)
translate_to_en_chain = TRANSLATE_TO_EN_TEMPLATE | llm | StrOutputParser()
translate_to_lang_chain = TRANSLATE_TO_LANG_TEMPLATE | llm | StrOutputParser()
classify_book_chain = CLASSIFY_BOOK_TEMPLATE | _llm_zero | StrOutputParser()
rag_chain = PROMPT_TEMPLATE | llm | StrOutputParser()

# Multi-query: LLM generates alternative phrasings for broader retrieval coverage
MULTI_QUERY_TEMPLATE = ChatPromptTemplate.from_template(
    "Generate 3 alternative phrasings of the following question for semantic document search.\n"
    "Output only the questions, one per line, no numbering or extra text.\n\n"
    "Question: {question}"
)
multi_query_chain = MULTI_QUERY_TEMPLATE | _llm_zero | StrOutputParser()


def format_chat_history(history: list, max_turns: int = 5) -> str:
    """
    Converts Gradio's messages-format history into a readable string for the prompt.

    Limits to the last max_turns exchanges to control token usage and keep
    context relevant. Older turns are dropped silently.

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.
        max_turns: Maximum number of user+assistant exchanges to include.

    Returns:
        A formatted string, or "No previous conversation." if history is empty.
    """
    if not history:
        return "No previous conversation."

    # Keep only the last N turns (each turn = 1 user + 1 assistant message)
    recent = history[-(max_turns * 2):]
    lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent]
    return "\n".join(lines)


def inference(query: str, history: list = None) -> tuple:
    """
    Run the RAG pipeline for a given query, with automatic language detection,
    conversation memory, multi-query retrieval, and FlashRank reranking.

    Steps:
        1. Detect the language of the query
        2. If not English, translate the query to English
        3. Classify which book the query targets
        4. Quick off-topic check (single-doc similarity score)
        5. Multi-query retrieval + FlashRank reranking
        6. Format conversation history
        7. Generate answer in English
        8. If original query was not English, translate answer back

    Args:
        query:   The user's question (any language).
        history: Gradio chat history (list of role/content dicts). Optional.

    Returns:
        A tuple of (answer string, usage dict).
    """

    # 1. Detect language using langdetect (statistical, no LLM call)
    lang_code = detect_lang(query)
    is_english = lang_code == "en"
    detected_language = LANGUAGE_NAMES.get(lang_code, lang_code.capitalize())

    with get_openai_callback() as cb:

        # 2. Translate query to English if needed
        english_query = (
            query
            if is_english
            else translate_to_en_chain.invoke({"text": query})
        )

        # 3. Classify which book the query targets
        raw_book = classify_book_chain.invoke({"query": english_query}).strip().lower()
        target_book = raw_book if raw_book in BOOK_KEYS else "all"
        print(f"  Book filter: {target_book}")

        # Build Chroma metadata filter
        chroma_filter = {"book": target_book} if target_book != "all" else None

        # 4. Quick off-topic check — single doc search, L2 distance (lower = more similar)
        preliminary = vectorstore.similarity_search_with_score(
            english_query, k=1, filter=chroma_filter
        )
        best_score = preliminary[0][1] if preliminary else float("inf")
        print(f"  Best similarity score: {best_score:.4f} (threshold: {RELEVANCE_THRESHOLD})")

        if best_score > RELEVANCE_THRESHOLD:
            off_topic_msg = (
                "I could not find relevant information about this topic in any of the books. "
                "Please ask a question related to Yuval Noah Harari's books: "
                "*Sapiens*, *Homo Deus*, or *21 Lessons for the 21st Century*."
            )
            if not is_english:
                off_topic_msg = translate_to_lang_chain.invoke({
                    "language": detected_language,
                    "text": off_topic_msg,
                })
            return off_topic_msg, {
                "tokens_input": cb.prompt_tokens,
                "tokens_output": cb.completion_tokens,
                "tokens_total": cb.total_tokens,
                "cost": cb.total_cost,
                "best_score": best_score,
                "off_topic": True,
            }

        # 5. Multi-query retrieval + FlashRank reranking
        #
        # MULTI-QUERY: The LLM rewrites the question in several ways (e.g. more formal,
        # more specific, from a different angle). Each variant is searched independently
        # in Chroma. Results are merged and deduplicated. This catches relevant chunks
        # that a single query phrasing might miss.
        #
        # RERANKING: A cross-encoder (FlashRank) re-scores every retrieved chunk against
        # the ORIGINAL query and keeps only the top_n most relevant. Cross-encoders are
        # more accurate than embedding similarity but too slow to run on the whole DB —
        # so we use them only on the small candidate set from multi-query retrieval.
        search_kwargs = {"k": RETRIEVAL_K * 2}  # retrieve more candidates before reranking
        if chroma_filter:
            search_kwargs["filter"] = chroma_filter

        # Multi-query: generate alternative phrasings and search each one
        variants_raw = multi_query_chain.invoke({"question": english_query})
        variants = [q.strip() for q in variants_raw.strip().split("\n") if q.strip()]
        all_queries = [english_query] + variants[:3]

        seen: set[str] = set()
        candidates = []
        for q in all_queries:
            docs = vectorstore.similarity_search(q, **search_kwargs)
            for doc in docs:
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    candidates.append(doc)

        # Rerank: cross-encoder rescores candidates and keeps top_n
        results = FlashrankRerank(top_n=RERANK_TOP_N).compress_documents(
            candidates, english_query
        )
        print(f"  Retrieved and reranked {len(results)} documents")

        # 6. Format conversation history for the prompt
        formatted_history = format_chat_history(history or [])

        # 7. Format context and generate answer in English
        context = "\n\n".join(doc.page_content for doc in results)
        english_response = rag_chain.invoke({
            "context": context,
            "query": english_query,
            "history": formatted_history,
        })

        # 8. Translate answer back to original language if needed
        final_response = (
            english_response
            if is_english
            else translate_to_lang_chain.invoke({
                "language": detected_language,
                "text": english_response,
            })
        )

    return final_response, {
        "tokens_input": cb.prompt_tokens,
        "tokens_output": cb.completion_tokens,
        "tokens_total": cb.total_tokens,
        "cost": cb.total_cost,
        "best_score": best_score,
        "off_topic": False,
    }


if __name__ == "__main__":
    test_query = "Who is Homo sapiens?"
    print(f"Query: {test_query}\n")
    answer, usage = inference(test_query)
    print(answer)
    print(f"\nTokens: {usage['tokens_total']} (in: {usage['tokens_input']} | out: {usage['tokens_output']}) — cost: ${usage['cost']:.4f}")
