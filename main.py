import streamlit as st

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()
import shutil
st.write("Deno:", shutil.which("deno"))


st.set_page_config(page_title="AI Video Assistant", page_icon="🎥", layout="wide")

# ---------- Session state ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def run_pipeline(source: str, language: str) -> dict:
    """Same logic as your original main.py, with progress feedback."""
    progress = st.progress(0, text="Reading input...")

    chunks = process_input(source)
    progress.progress(20, text="Transcribing audio...")

    transcript = transcribe_all(chunks, language=language)
    progress.progress(50, text="Generating title...")

    title = generate_title(transcript)
    progress.progress(60, text="Summarizing...")

    summary = summarize(transcript)
    progress.progress(70, text="Extracting action items...")

    action_items = extract_action_items(transcript)
    progress.progress(80, text="Extracting key decisions...")

    decisions = extract_key_decisions(transcript)
    progress.progress(85, text="Extracting open questions...")

    questions = extract_questions(transcript)
    progress.progress(95, text="Building chat index...")

    rag_chain = build_rag_chain(transcript)
    progress.progress(100, text="Done")
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------- Sidebar: input form ----------
with st.sidebar:
    st.header("🎥 AI Video Assistant")
    source = st.text_input("YouTube URL or local file path")
    language = st.selectbox("Language", ["english", "hinglish"])
    run_clicked = st.button("Process", type="primary", use_container_width=True)

    if run_clicked:
        if not source.strip():
            st.error("Enter a YouTube URL or file path first.")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    st.session_state.result = run_pipeline(source.strip(), language)
                    st.session_state.chat_history = []
                    st.success("Processed successfully.")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

# ---------- Main area ----------
result = st.session_state.result

if result is None:
    st.info("Enter a source in the sidebar and click **Process** to get started.")
else:
    st.title(result["title"])

    tab_summary, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary & Insights", "📝 Full Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.subheader("Summary")
        st.write(result["summary"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("✅ Action Items")
            st.write(result["action_items"])
        with col2:
            st.subheader("🔑 Key Decisions")
            st.write(result["key_decisions"])
        with col3:
            st.subheader("❓ Open Questions")
            st.write(result["open_questions"])

    with tab_transcript:
        st.subheader("Full Transcript")
        st.text_area("Transcript", result["transcript"], height=500, label_visibility="collapsed")

    with tab_chat:
        st.subheader("Chat with your meeting")

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        question = st.chat_input("Ask a question about this video...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Error: {e}"
                    st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))