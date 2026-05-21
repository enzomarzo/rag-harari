"""
app.py — Gradio chat interface

Wraps the inference pipeline in a Gradio ChatInterface.
Make sure you have run ingestion.py at least once before starting the app.

Usage:
    python app.py
"""

import json
import os
import gradio as gr
from inference import inference as run_inference
from config import BOOKS

# Short tab labels per book key
BOOK_TAB_LABELS = {
    "sapiens": "Sapiens",
    "homo_deus": "Homo Deus",
    "21_lessons": "21 Lessons for the XXI",
}


def _load_index() -> dict:
    """Load book_index.json and return a dict keyed by book key.

    The file currently stores a flat list of parts for Sapiens.
    Future versions may store {book_key: [parts]} directly.
    """
    path = "book_index.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Flat list → treat as Sapiens index (legacy format)
    if isinstance(data, list):
        return {"sapiens": data}
    return data


BOOK_INDEX = _load_index()


def chat_fn(message: str, history: list) -> str:
    # Strip flag prefix if the user clicked an example button (e.g. "🇬🇧 Who is...")
    for prefix in ["🇬🇧 ", "🇵🇹 "]:
        if message.startswith(prefix):
            message = message[len(prefix):]
            break

    answer, usage = run_inference(message, history)

    footnote = (
        f"\n\n---\n"
        f"*Tokens: {usage['tokens_total']:,} "
        f"(in: {usage['tokens_input']:,} | out: {usage['tokens_output']:,})"
        f" — est. cost: ${usage['cost']:.4f}"
        f" — initial score: {usage['best_score']:.3f}*"
    )

    return answer + footnote


demo = gr.Blocks(
    title="Yuval Noah Harari — RAG Assistant",
    css="""
    #back-link {
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        color: #888 !important;
        font-size: 0.8em !important;
        padding: 2px 0 !important;
        min-width: unset !important;
        width: auto !important;
        display: inline-block;
    }
    #back-link:hover { color: #555 !important; }
    """,
)

with demo:
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chat = gr.ChatInterface(
                fn=chat_fn,
                title="Yuval Noah Harari — RAG Assistant",
                description=(
                    "Ask questions about **Sapiens**, **Homo Deus**, or **21 Lessons for the 21st Century** "
                    "by Yuval Noah Harari. Answers are grounded in the books. "
                    "You can ask in any language."
                ),
                examples=[
                    "🇬🇧 Who is Homo sapiens?",
                    "🇬🇧 What does Harari predict about the future of humanity in Homo Deus?",
                    "🇬🇧 What are the most important challenges of the 21st century?",
                    "🇬🇧 How did agriculture change human society?",
                    "🇵🇹 O que é a Revolução Cognitiva?",
                    "🇵🇹 O que o Homo Deus diz sobre inteligência artificial?",
                ],
            )

            back_btn = gr.Button("← Back to examples", elem_id="back-link")
            back_btn.click(fn=lambda: [], outputs=chat.chatbot)

        # ------------------------------------------------------------------
        # Book index tab — one sub-tab per book, with language toggle
        # ------------------------------------------------------------------
        with gr.Tab("📖 Book Index"):
            lang_toggle = gr.Radio(
                choices=["🇧🇷 Português", "🇬🇧 English"],
                value="🇧🇷 Português",
                label="Summary language",
                interactive=True,
            )

            index_mds: list[gr.Markdown] = []
            index_content: list[dict] = []  # {"en": str, "pt": str} per accordion

            with gr.Tabs():
                for book in BOOKS:
                    key = book["book"]
                    label = BOOK_TAB_LABELS.get(key, book["title"])
                    parts = BOOK_INDEX.get(key)
                    with gr.Tab(label):
                        if parts:
                            for part_data in parts:
                                content_en = "\n\n".join(
                                    f"#### {ch['title']}\n{ch.get('summary_en', ch.get('summary', ''))}"
                                    for ch in part_data["chapters"]
                                )
                                content_pt = "\n\n".join(
                                    f"#### {ch['title']}\n{ch.get('summary_pt', ch.get('summary_en', ch.get('summary', '')))}"
                                    for ch in part_data["chapters"]
                                )
                                index_content.append({"en": content_en, "pt": content_pt})
                                with gr.Accordion(part_data["part"], open=False):
                                    index_mds.append(gr.Markdown(content_pt))
                        else:
                            gr.Markdown(
                                f"_Index not yet generated for **{book['title']}**._\n\n"
                                "Run `python generate_index.py` to build it."
                            )

            if index_mds:
                _captured_content = index_content

                def _switch_lang(lang, content=_captured_content):
                    is_pt = "Português" in lang
                    return [c["pt"] if is_pt else c["en"] for c in content]

                lang_toggle.change(fn=_switch_lang, inputs=lang_toggle, outputs=index_mds)

if __name__ == "__main__":
    demo.launch(debug=True)
