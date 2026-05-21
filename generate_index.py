"""
generate_index.py — Generate a chapter-by-chapter summary index for all three books.

For each chapter, retrieves relevant chunks from Chroma (filtered by book) and
generates a 2-sentence summary using the LLM. Saves to book_index.json.

Run once after ingestion:
    python generate_index.py

Already-indexed books are skipped automatically.
To regenerate a specific book, remove its key from book_index.json and run again.
To regenerate everything, delete book_index.json and run again.
"""

import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL

load_dotenv()

OUTPUT_FILE = "book_index.json"

# ---------------------------------------------------------------------------
# Chapter structure for all three books
# ---------------------------------------------------------------------------
BOOKS_CHAPTERS = {
    "sapiens": {
        "title": "Sapiens: A Brief History of Humankind",
        "parts": {
            "Part One — The Cognitive Revolution": [
                "An Animal of No Significance",
                "The Tree of Knowledge",
                "A Day in the Life of Adam and Eve",
                "The Flood",
            ],
            "Part Two — The Agricultural Revolution": [
                "History's Biggest Fraud",
                "Building Pyramids",
                "Memory Overload",
                "There is No Justice in History",
            ],
            "Part Three — The Unification of Humankind": [
                "The Arrow of History",
                "The Scent of Money",
                "Imperial Visions",
                "The Law of Religion",
                "The Secret of Success",
            ],
            "Part Four — The Scientific Revolution": [
                "The Discovery of Ignorance",
                "The Marriage of Science and Empire",
                "The Capitalist Creed",
                "The Wheels of Industry",
                "A Permanent Revolution",
                "And They Lived Happily Ever After",
                "The End of Homo Sapiens",
            ],
        },
    },
    "homo_deus": {
        "title": "Homo Deus: A Brief History of Tomorrow",
        "parts": {
            "Part One — Homo Sapiens Conquers the World": [
                "The New Human Agenda",
                "The Anthropocene",
            ],
            "Part Two — Homo Sapiens Gives Meaning to the World": [
                "The Human Spark",
                "The Storytellers",
                "The Odd Couple",
                "The Modern Covenant",
                "The Humanist Revolution",
            ],
            "Part Three — Homo Sapiens Loses Control": [
                "The Time Bomb in the Laboratory",
                "The Great Decoupling",
                "The Ocean of Consciousness",
                "The Data Religion",
            ],
        },
    },
    "21_lessons": {
        "title": "21 Lessons for the 21st Century",
        "parts": {
            "Part I — The Technological Challenge": [
                "Disillusionment",
                "Work",
                "Liberty",
                "Equality",
            ],
            "Part II — The Political Challenge": [
                "Community",
                "Civilization",
                "Nationalism",
                "Religion",
                "Immigration",
            ],
            "Part III — Despair and Hope": [
                "Terrorism",
                "War",
                "Humility",
                "God",
                "Secularism",
            ],
            "Part IV — Truth": [
                "Ignorance",
                "Justice",
                "Post-Truth",
                "Science Fiction",
            ],
            "Part V — Resilience": [
                "Education",
                "Meaning",
                "Meditation",
            ],
        },
    },
}

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Based on the following excerpts from the book "{book_title}" by Yuval Noah Harari,
write a concise 2-sentence summary of the chapter titled "{chapter}".
Focus on the main idea. Be factual and grounded in the text.

Excerpts:
{context}

Summary:"""
)

TRANSLATE_PT_PROMPT = ChatPromptTemplate.from_template(
    "Translate the following text to Brazilian Portuguese. "
    "Reply with only the translated text, nothing else.\n\nText: {text}"
)


def _load_existing() -> dict:
    """Load existing index, migrating from legacy flat-list format if needed."""
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):  # legacy: flat list was Sapiens only
        return {"sapiens": data}
    return data


def _migrate_translations(index: dict, translate_chain) -> bool:
    """
    Rename legacy 'summary' key to 'summary_en' and add 'summary_pt' where missing.
    Returns True if any change was made.
    """
    changed = False
    for book_key, parts in index.items():
        for part_data in parts:
            for ch in part_data["chapters"]:
                # Rename legacy key
                if "summary" in ch and "summary_en" not in ch:
                    ch["summary_en"] = ch.pop("summary")
                    changed = True
                # Add PT if missing
                if "summary_pt" not in ch and "summary_en" in ch:
                    ch["summary_pt"] = translate_chain.invoke(
                        {"text": ch["summary_en"]}
                    ).strip()
                    print(f"    🇧🇷 translated: {ch['title']}")
                    changed = True
    return changed


def generate_index():
    index = _load_existing()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    chain_summary = SUMMARY_PROMPT | llm | StrOutputParser()
    chain_translate_pt = TRANSLATE_PT_PROMPT | llm | StrOutputParser()

    # --- migrate existing entries that lack summary_pt ---
    print("Checking for untranslated summaries...")
    if _migrate_translations(index, chain_translate_pt):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print("✓ Translations added to existing entries.\n")
    else:
        print("✓ All existing entries already have translations.\n")

    any_generated = False

    for book_key, book_data in BOOKS_CHAPTERS.items():
        if book_key in index:
            print(f"✓ {book_data['title']} — already indexed, skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"GENERATING INDEX: {book_data['title']}")
        print("=" * 80)
        any_generated = True

        book_parts = []
        for part, chapters in book_data["parts"].items():
            print(f"\n  {part}")
            part_data = {"part": part, "chapters": []}

            for chapter in chapters:
                query = f"{chapter} — {book_data['title']}"
                results = vectorstore.similarity_search(
                    query, k=4, filter={"book": book_key}
                )
                context = "\n\n".join(doc.page_content for doc in results)

                summary_en = chain_summary.invoke({
                    "book_title": book_data["title"],
                    "chapter": chapter,
                    "context": context,
                }).strip()

                summary_pt = chain_translate_pt.invoke(
                    {"text": summary_en}
                ).strip()

                print(f"    ✓ {chapter}")

                part_data["chapters"].append({
                    "title": chapter,
                    "summary_en": summary_en,
                    "summary_pt": summary_pt,
                })

            book_parts.append(part_data)

        index[book_key] = book_parts

        # Save incrementally after each book
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Saved {book_data['title']} to '{OUTPUT_FILE}'")

    if not any_generated:
        print("\nAll books already indexed. Delete book_index.json to regenerate.")
    else:
        print("\n✓ Done.")

    if not any_generated:
        print("\nAll books already indexed. Delete book_index.json to regenerate.")
    else:
        print("\n✓ Done.")


if __name__ == "__main__":
    generate_index()
