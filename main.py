
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="VideoMind AI",
    page_icon="🎥",
    layout="wide"
)

load_dotenv()

# ---------- Custom Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --burgundy: #800020;
    --cream: #FAF7F2;
    --ink: #292321;
    --muted: #77716D;
}

.stApp {
    background-color: var(--cream);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background-color: #F0EAE2;
    border-right: 1px solid #E5DCD2;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--burgundy);
}

.stButton > button {
    background-color: var(--burgundy);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #5F0018;
    color: white;
    border: none;
}

.stTextInput input,
.stSelectbox [data-baseweb="select"] {
    border-radius: 9px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E9E1D8;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--burgundy);
}

hr {
    border-color: #E5DCD2;
}

.hero {
    background: linear-gradient(120deg, #800020, #4B1021);
    padding: 38px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white !important;
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.hero p {
    color: #F4DDE3;
    font-size: 1rem;
}

.section-label {
    color: #800020;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.insight-card {
    background: white;
    border: 1px solid #E9E1D8;
    border-radius: 14px;
    padding: 20px;
    min-height: 160px;
}

.insight-card h3 {
    font-size: 1.1rem;
    margin-top: 0;
}

.demo-card {
    background: white;
    border: 1px solid #E9E1D8;
    border-radius: 14px;
    padding: 20px;
}

.small-muted {
    color: #77716D;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Session State ----------
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- Processing Pipeline ----------
def run_pipeline(source: str, language: str) -> dict:

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


# ---------- Sidebar ----------
with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 30px;">🎥</div>
            <h2 style="color:#800020; margin-bottom:0;">
                VideoMind AI
            </h2>
            <p style="color:#77716D; font-size:13px;">
                Your intelligent video companion
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Add your source")

    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="Paste your video link...",
        label_visibility="collapsed"
    )

    language = st.selectbox(
        "Transcript language",
        ["english", "hinglish"]
    )

    run_clicked = st.button(
        "✨  Process Video",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption("SUPPORTED FEATURES")

    st.markdown("""
    - 📝 Automatic transcription
    - ✨ AI-generated summaries
    - ✅ Action item extraction
    - 🔑 Key decision identification
    - 💬 Chat with your video
    """)

    st.divider()

    st.caption("Built with Python · Streamlit · AI")

    if run_clicked:

        if not source.strip():
            st.error("Please enter a video URL or file path.")

        else:
            try:
                with st.spinner("Processing your video..."):
                    st.session_state.result = run_pipeline(
                        source.strip(),
                        language
                    )

                st.session_state.chat_history = []
                st.success("Video processed successfully!")

            except Exception as e:
                st.error(f"Pipeline failed: {e}")


# ---------- Main Dashboard ----------
result = st.session_state.result

# ---------- Welcome Screen ----------
if result is None:

    st.markdown("""
    <div class="hero">
        <div class="section-label" style="color:#F4DDE3;">
            AI-POWERED VIDEO INTELLIGENCE
        </div>
        <h1>Understand more.<br>Spend less time watching.</h1>
        <p>
            Transform long videos and recordings into structured,
            searchable knowledge with AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Everything you need, in one place")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="insight-card">
            <h3>📝 Smart Summaries</h3>
            <p class="small-muted">
                Get concise summaries of long videos and meetings.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="insight-card">
            <h3>✅ Actionable Insights</h3>
            <p class="small-muted">
                Identify tasks, decisions and unanswered questions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="insight-card">
            <h3>💬 Interactive AI Chat</h3>
            <p class="small-muted">
                Ask questions and retrieve information from your video.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### See it in action")

    st.markdown(
        '<div class="demo-card">',
        unsafe_allow_html=True
    )

    st.video("https://youtu.be/muYl234no_E")

    st.markdown(
        '<p class="small-muted">Watch the AI Video Assistant demo.</p></div>',
        unsafe_allow_html=True
    )

    st.info("Add a YouTube URL or local file path in the sidebar to get started.")


# ---------- Results Dashboard ----------
else:

    st.markdown(
        '<div class="section-label">YOUR VIDEO ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.title(result["title"])

    st.caption("Your video has been transcribed and analysed.")

    tab_summary, tab_transcript, tab_chat, tab_demo = st.tabs(
        [
            "📋  Summary & Insights",
            "📝  Transcript",
            "💬  Ask AI",
            "🎬  Project Demo"
        ]
    )

    # ---------- Summary ----------
    with tab_summary:

        st.markdown("### Overview")

        with st.container(border=True):
            st.markdown("#### ✨ AI Summary")
            st.write(result["summary"])

        st.markdown("### Key Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown("#### ✅ Action Items")
                st.write(result["action_items"])

        with col2:
            with st.container(border=True):
                st.markdown("#### 🔑 Key Decisions")
                st.write(result["key_decisions"])

        with col3:
            with st.container(border=True):
                st.markdown("#### ❓ Open Questions")
                st.write(result["open_questions"])

    # ---------- Transcript ----------
    with tab_transcript:

        st.markdown("### Full Transcript")

        st.caption("Complete transcription of your video.")

        st.text_area(
            "Transcript",
            result["transcript"],
            height=500,
            label_visibility="collapsed"
        )

        st.download_button(
            "⬇ Download Transcript",
            data=result["transcript"],
            file_name="video_transcript.txt",
            mime="text/plain"
        )

    # ---------- Chat ----------
    with tab_chat:

        st.markdown("### Chat with your video")

        st.caption(
            "Ask questions about the content of your video."
        )

        for role, msg in st.session_state.chat_history:

            with st.chat_message(role):
                st.write(msg)

        question = st.chat_input(
            "Ask something about this video..."
        )

        if question:

            st.session_state.chat_history.append(
                ("user", question)
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:
                        answer = ask_question(
                            result["rag_chain"],
                            question
                        )

                    except Exception as e:
                        answer = f"Error: {e}"

                    st.write(answer)

            st.session_state.chat_history.append(
                ("assistant", answer)
            )

    # ---------- Demo ----------
    with tab_demo:

        st.markdown("### Project Demonstration")

        st.caption(
            "Watch a walkthrough of the AI Video Assistant."
        )

        st.video("https://youtu.be/muYl234no_E")
