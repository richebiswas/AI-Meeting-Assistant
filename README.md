# 🎙️ AI Meeting Assistant & Video Agent

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20LCEL-green.svg)](https://python.langchain.com/)
[![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper%20%7C%20Sarvam%20AI-orange.svg)](https://github.com/openai/whisper)
[![LLM](https://img.shields.io/badge/LLM-Gemini%203.6%20Flash%20%7C%20Groq-purple.svg)](https://groq.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-red.svg)](https://www.trychroma.com/)

An end-to-end intelligent meeting analysis and retrieval system that processes audio files and YouTube videos into structured meeting minutes, actionable insights, and an interactive **Retrieval-Augmented Generation (RAG)** chat assistant.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [1. CLI Pipeline & Interactive Chat](#1-running-the-cli-pipeline)
  - [2. Streamlit Web Interface](#2-running-the-streamlit-ui)
- [How It Works Under the Hood](#-how-it-works-under-the-hood)
  - [Audio Pre-processing](#audio-pre-processing)
  - [Dual-Engine Speech-to-Text](#dual-engine-speech-to-text)
  - [Map-Reduce Summarization & Insights](#map-reduce-summarization--insights)
  - [RAG & ChromaDB Vector Store](#rag--chromadb-vector-store)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 Overview

Long meetings and video lectures create hours of unstructured audio that are difficult to review. The **AI Meeting Assistant** automates this workflow:
1. Downloads audio from YouTube URLs or ingests local media files.
2. Standardizes audio into **16 kHz Mono WAV** and slices it into manageable 10-minute chunks.
3. Transcribes speech accurately using **OpenAI Whisper** (English) or **Sarvam AI** (Hinglish / Indic code-switching).
4. Generates professional titles, Map-Reduce bulleted summaries, and structured outputs (Action Items, Key Decisions, Open Questions) via **Gemini 3.6 Flash**.
5. Embeds transcripts locally into a persistent **ChromaDB** vector store using **HuggingFace `all-MiniLM-L6-v2`**.
6. Delivers ultra-low-latency, grounded conversational Q&A over the meeting context powered by **Groq**.

---

## ✨ Key Features

- **Multi-Source Ingestion**: Paste any YouTube URL or provide a local audio/video file path.
- **Audio Standardization**: Automatic conversion to 16 kHz Mono WAV using `yt-dlp`, `pydub`, and FFmpeg.
- **Dual STT Engine**:
  - **Local Whisper (`small`)**: Fast, offline, private, and zero API cost for English audio.
  - **Sarvam AI (`saaras:v4`)**: Specialized transcription and translation for Hinglish code-mixed discussions.
- **Long-Form Audio Chunking**: Automatically segments long recordings into 10-minute windows (and 25s micro-pieces for Sarvam) to avoid memory crashes and token limits.
- **Map-Reduce Summarization**: Overcomes context window bottlenecks by summarizing chunks in parallel before synthesizing the final executive summary.
- **Actionable Insight Extraction**: Automatically identifies:
  - ✅ **Action Items** (with assigned owners and deadlines)
  - 🔑 **Key Decisions**
  - ❓ **Open Questions & Follow-ups**
- **Strictly Grounded RAG Chat**: Ask specific questions about the meeting. System prompts prevent hallucination by strictly restricting answers to the transcript context.
- **Decoupled Architecture**: Clean separation between core AI logic, media utilities, execution pipelines, and UI layers.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Input: YouTube URL or Local Audio File] --> B[utils/audio_processor.py]
    B -->|yt-dlp + FFmpeg| C[16 kHz Mono WAV]
    C -->|Pydub| D[10-Minute Audio Chunks]
    D --> E[core/transcriber.py]
    
    subgraph STT Engine Selection
        E -->|Language = English| F[Local Whisper - model: small]
        E -->|Language = Hinglish| G[Sarvam AI API - model: saaras:v4]
    end
    
    F --> H[Full Text Transcript]
    G --> H
    
    H --> I[pipeline.py Orchestrator]
    
    I --> J[core/summarize.py]
    J -->|Gemini 3.6 Flash| K[Title & Map-Reduce Summary]
    
    I --> L[core/extractor.py]
    L -->|Gemini 3.6 Flash| M[Action Items, Decisions, Questions]
    
    I --> N[core/vector_store.py]
    N -->|all-MiniLM-L6-v2 Embeddings| O[ChromaDB Vector Store]
    
    O --> P[core/rag_engine.py]
    P -->|Groq API: gpt-oss-20b| Q[Interactive RAG Chat Chain]
📂 Project Structure
text


Video Agent/
├── core/
│   ├── extractor.py         # Extracts Action Items, Decisions, and Open Questions
│   ├── rag_engine.py        # LCEL RAG chain assembly and Groq Q&A engine
│   ├── summarize.py         # Title generation & Map-Reduce summarization
│   ├── transcriber.py       # Speech-to-Text router (Local Whisper / Sarvam AI)
│   └── vector_store.py      # ChromaDB setup and HuggingFace embedding indexing
├── utils/
│   └── audio_processor.py   # YouTube download, 16 kHz mono conversion, chunking
├── downloads/               # Directory for downloaded audio and processed chunks
├── vector_db/               # Persistent ChromaDB sqlite3 database and vector index
├── pipeline.py              # Headless pipeline orchestrator & interactive CLI
├── main.py                  # Streamlit web UI application
├── packages.txt             # Linux system dependencies (ffmpeg)
├── requirements.txt         # Python package dependencies
├── .env                     # API keys and environment variables (ignored in git)
└── README.md                # Project documentation
🛠️ Tech Stack
Category	Tools & Libraries
Language	Python 3.10+ / 3.11
Orchestration & LCEL	LangChain Core, LangChain Community
Audio Processing	FFmpeg, pydub, yt-dlp
Speech-to-Text (STT)	OpenAI Whisper (openai-whisper), Sarvam AI SDK (sarvamai)
Generative LLMs	Google Gemini 3.6 Flash (langchain-google-genai), Groq (langchain-groq)
Embeddings	HuggingFace sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
Vector Database	ChromaDB (chromadb, langchain-chroma)
User Interface	Streamlit (CLI & upcoming FastAPI backend)
📋 Prerequisites
Python 3.10 or 3.11 installed.
FFmpeg installed and accessible in your system's PATH.
Windows: Install via winget install Gyan.FFmpeg or choco install ffmpeg.
macOS: brew install ffmpeg
Linux/Ubuntu: sudo apt update && sudo apt install -y ffmpeg
API Keys:
Google Gemini API Key (Required for summarization and extraction)
Groq API Key (Required for RAG chat responses)
Sarvam AI API Key (Optional, required if processing Hinglish audio)
⚙️ Installation & Setup
Clone the repository:

bash


git clone https://github.com/richebiswas/AI-Meeting-Assistant.git
cd "Video Agent"
Create and activate a virtual environment:

bash


# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

bash


pip install --upgrade pip
pip install -r requirements.txt
Configure Environment Variables: Create a .env file in the root directory:

env


# LLM API Keys
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
# Sarvam AI (for Hinglish STT)
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_MODEL=saaras:v4
# Whisper Configuration (Optional: tiny, base, small, medium, large)
WHISPER_MODEL=small
🖥️ Usage
1. Running the CLI Pipeline
For automated headless execution or terminal-based chat:

bash


python pipeline.py
Workflow:

Enter a YouTube URL (e.g. https://www.youtube.com/watch?v=...) or a local audio path (e.g. sample.mp3).
Choose language: english (default) or hinglish.
The pipeline will download, process, transcribe, summarize, and display all insights in the terminal.
You will immediately enter an interactive chat loop (You:) to ask questions grounded directly in the meeting transcript. Type exit or q to quit.
2. Running the Streamlit UI
To launch the graphical web dashboard:

bash


streamlit run main.py
Features in the UI:

Sidebar: Input video/audio source, select language, and monitor processing progress.
Tab 1: 📋 Summary & Insights: Title, executive summary, action items, key decisions, and open questions.
Tab 2: 📝 Full Transcript: Complete raw transcript with review and copy options.
Tab 3: 💬 Chat: Multi-turn chat interface to query the meeting using RAG.
🔬 How It Works Under the Hood
Audio Pre-processing
Incoming audio is re-sampled to 16,000 Hz and downmixed to a single channel (mono).
Why 16 kHz Mono?
Speech frequencies fall well below 8 kHz. By the Nyquist-Shannon sampling theorem, 16 kHz captures all vocal harmonics while eliminating unnecessary high-frequency noise.
Whisper's internal log-Mel spectrogram feature extractor natively expects 16 kHz single-channel input. Pre-converting minimizes runtime memory spikes and CPU compute.
Files are sliced into 10-minute segments using pydub to prevent Out-Of-Memory (OOM) errors.
Dual-Engine Speech-to-Text
English: Processed locally via OpenAI Whisper (small model). Zero API costs, complete data privacy, and robust accuracy in noisy environments.
Hinglish: Sliced into 25-second micro-pieces and dispatched to Sarvam AI's saaras:v4 model in mode="translate", which accurately converts mixed Hindi-English colloquial audio into clear English text.
Map-Reduce Summarization & Insights
Transcripts exceeding prompt context limits are chunked via RecursiveCharacterTextSplitter (3,000 characters, 200 overlap).
Map Phase: Each chunk is summarized individually by Gemini 3.6 Flash.
Reduce Phase: Partial summaries are consolidated into a structured, bulleted executive summary.
Dedicated extraction chains isolate Action Items, Key Decisions, and Unresolved Questions with zero cross-task prompt dilution.
RAG & ChromaDB Vector Store
Transcripts are split into small 500-character chunks with a 50-character overlap.
Each chunk is embedded locally on CPU using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) and stored in ChromaDB (vector_db/).
When a user asks a question:
The question is embedded and compared against stored vectors using similarity search (
k
=
4
k=4).
The top 4 chunks are formatted into a context payload.
Groq's high-speed inference engine (openai/gpt-oss-20b) formulates an answer strictly from the retrieved context.
🗺️ Roadmap
 FastAPI Backend: Migrate the orchestration layer into asynchronous REST and WebSocket endpoints.
 Modern Frontend: Build a responsive HTML/CSS/JavaScript or React web application.
 Speaker Diarization: Integrate pyannote.audio to tag speaker identities ([Speaker 1], [Speaker 2]).
 Voice Activity Detection (VAD): Replace fixed 10-minute audio slicing with dynamic silence detection (Silero VAD).
 Export to PDF/Notion: One-click export of formatted meeting summaries and action items.
📄 License
This project is open-source under the 
MIT License
. Feel free to use, modify, and distribute it for personal or commercial projects.

8:03 PM
