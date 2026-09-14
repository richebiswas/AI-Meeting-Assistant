# AI Meeting Assistant

An AI-powered meeting assistant that converts meeting or video audio into a structured, searchable knowledge base.

The application processes audio, generates a transcript, summarizes the discussion, extracts important information such as action items and key decisions, and allows users to ask questions about the meeting using Retrieval-Augmented Generation (RAG).

> **Current status:** The core Streamlit application and AI pipeline are implemented. A FastAPI backend and HTML/CSS/JavaScript frontend are currently being developed to move the project toward a more production-style architecture.

---

## Features

* 🎙️ **Audio Processing**

  * Accepts YouTube URLs or local audio files
  * Extracts and processes audio using FFmpeg
  * Converts audio to mono, 16 kHz WAV format
  * Splits long audio into manageable chunks

* 📝 **Speech-to-Text**

  * Local transcription using OpenAI Whisper
  * Hinglish transcription/translation support using Sarvam AI

* 🤖 **AI Meeting Analysis**

  * Generates a meeting title
  * Produces an AI-generated summary
  * Extracts action items
  * Extracts key decisions
  * Identifies open questions

* 🔎 **RAG-based Meeting Q&A**

  * Converts transcript chunks into embeddings
  * Stores vectors using ChromaDB
  * Retrieves relevant transcript sections
  * Uses an LLM to answer questions based on the meeting transcript

* 💬 **Interactive UI**

  * Streamlit interface for processing meetings
  * Transcript, summary, and insights
  * Conversational Q&A over the meeting

* 📄 **Export**

  * Export generated meeting information to TXT/PDF

---

## Architecture

```text
                    ┌──────────────────────┐
                    │   YouTube URL /      │
                    │   Local Audio File   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Audio Processing   │
                    │   yt-dlp + FFmpeg    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Audio Conversion   │
                    │   Mono + 16 kHz WAV  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Audio Chunking     │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Speech Recognition        │
              │                                 │
              │ Whisper       │      Sarvam AI  │
              │ English       │      Hinglish   │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Transcript       │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌────────────┐    ┌────────────┐    ┌────────────┐
      │  Summary   │    │  Actions   │    │ Decisions  │
      └────────────┘    └────────────┘    └────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     RAG Pipeline     │
                    │                      │
                    │ HuggingFace         │
                    │ Embeddings           │
                    │        ↓             │
                    │ ChromaDB             │
                    │        ↓             │
                    │ Retriever            │
                    │        ↓             │
                    │ Groq LLM             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Meeting Q&A        │
                    └──────────────────────┘
```

---

## Project Structure

```text
AI-Meeting-Assistant/
│
├── core/
│   ├── transcriber.py
│   ├── summarize.py
│   ├── extractor.py
│   ├── rag_engine.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── main.py
├── pipeline.py
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

### Main Components

#### `pipeline.py`

Acts as the main orchestration layer.

It connects the different stages of the application:

```text
Audio Input
    ↓
Audio Processing
    ↓
Transcription
    ↓
Title Generation
    ↓
Summarization
    ↓
Information Extraction
    ↓
RAG Construction
```

The pipeline returns the generated meeting information and RAG chain to the application.

#### `main.py`

Contains the current Streamlit interface.

It provides the user-facing interaction for:

* Providing an audio/video source
* Selecting the language
* Processing the meeting
* Viewing the transcript
* Viewing the summary and extracted information
* Asking questions about the meeting

#### `utils/audio_processor.py`

Responsible for preparing audio before transcription.

The processing flow is:

```text
Input
 ↓
YouTube download / local file
 ↓
FFmpeg conversion
 ↓
Mono + 16 kHz WAV
 ↓
Audio chunking
 ↓
Transcription-ready chunks
```

#### `core/transcriber.py`

Handles speech-to-text.

Two transcription paths are supported:

* **Whisper** for English/local transcription
* **Sarvam AI** for Hinglish processing and translation

#### `core/summarize.py`

Uses Gemini to generate:

* Meeting title
* Meeting summary

#### `core/extractor.py`

Uses an LLM to extract structured information from the transcript:

* Action items
* Key decisions
* Open questions

#### `core/vector_store.py`

Creates vector representations of transcript chunks using Hugging Face embeddings and stores them in ChromaDB.

#### `core/rag_engine.py`

Builds the RAG question-answering pipeline.

The basic flow is:

```text
User Question
      ↓
Retriever
      ↓
Relevant Transcript Chunks
      ↓
Prompt + Context
      ↓
Groq LLM
      ↓
Answer
```

---

## Tech Stack

### Programming Language

* Python

### Audio & Video Processing

* FFmpeg
* yt-dlp
* Pydub

### Speech Recognition

* OpenAI Whisper
* Sarvam AI

### LLM & GenAI

* Google Gemini
* Groq

### LLM Framework

* LangChain
* LangChain Expression Language (LCEL)

### RAG

* ChromaDB
* Hugging Face Sentence Transformers
* LangChain Chroma
* LangChain HuggingFace

### Application

* Streamlit

### Export

* ReportLab

### Environment Management

* Python-dotenv

---

## How the RAG System Works

The meeting transcript is divided into smaller chunks before being converted into vector embeddings.

```text
Transcript
    ↓
Text Chunks
    ↓
HuggingFace Embeddings
    ↓
Vector Representations
    ↓
ChromaDB
```

When the user asks a question:

```text
Question
   ↓
Embedding
   ↓
Similarity Search
   ↓
Top Relevant Transcript Chunks
   ↓
Context + Question
   ↓
Groq LLM
   ↓
Answer
```

This allows the assistant to answer questions using information contained in the meeting rather than relying only on the LLM's general knowledge.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/richebiswas/AI-Meeting-Assistant.git
cd AI-Meeting-Assistant
```

### 2. Create a virtual environment

Using Python 3.11 is recommended.

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is required for audio extraction and conversion.

Verify the installation:

```bash
ffmpeg -version
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_MODEL=saaras:v4
```

Never commit API keys or `.env` files to GitHub.

---

## Running the Application

### Streamlit Application

Run:

```bash
streamlit run main.py
```

The application will open in your browser.

### Pipeline / CLI

The backend pipeline can also be executed directly:

```bash
python pipeline.py
```

You will be prompted for:

```text
Enter YouTube URL or local file path:
Language (english/hinglish):
```

The pipeline then processes the audio and generates the meeting insights.

---

## Example Workflow

A typical meeting goes through the following process:

```text
Meeting Audio
     ↓
Audio preprocessing
     ↓
Whisper / Sarvam transcription
     ↓
Full transcript
     ↓
Gemini analysis
     ├── Title
     ├── Summary
     ├── Action Items
     ├── Key Decisions
     └── Open Questions
     ↓
Transcript embeddings
     ↓
ChromaDB
     ↓
RAG
     ↓
Interactive meeting Q&A
```

Example questions:

```text
What were the main decisions made?

Who was assigned the database task?

What problems were discussed?

What are the unresolved questions?

What was the main objective of the meeting?
```

---

## Why Mono 16 kHz Audio?

The audio is converted to:

```text
Mono
16,000 Hz sample rate
```

This is a practical format for speech-recognition models such as Whisper.

### Mono

A meeting recording usually does not require separate left and right channels for transcription.

Converting to mono:

* Reduces unnecessary audio data
* Simplifies processing
* Provides a consistent input format

### 16 kHz

16 kHz is commonly used for speech-processing pipelines and provides sufficient frequency information for speech recognition while keeping the data relatively lightweight.

---

## Current Limitations

* YouTube media downloads may fail in some cloud deployment environments because of external HTTP/CDN restrictions.
* Local audio processing works independently of YouTube access.
* LLM availability and API rate limits can affect title generation, summarization, and extraction.
* Local Whisper transcription can be computationally expensive depending on the selected model.
* Long meetings require chunking, which adds processing time.
* The current production architecture is still being developed.

---

## Roadmap

### Completed

* [x] Audio input processing
* [x] YouTube/local audio support
* [x] FFmpeg audio conversion
* [x] Audio chunking
* [x] Whisper transcription
* [x] Hinglish transcription/translation
* [x] AI-generated summaries
* [x] Action-item extraction
* [x] Key-decision extraction
* [x] Open-question extraction
* [x] ChromaDB vector storage
* [x] RAG-based transcript Q&A
* [x] Streamlit interface
* [x] PDF/TXT export
* [x] GitHub repository

### In Progress

* [ ] FastAPI backend
* [ ] HTML/CSS/JavaScript frontend
* [ ] Separation of frontend and backend
* [ ] Production-style API architecture
* [ ] Cloud deployment of the FastAPI backend
* [ ] Improved error handling
* [ ] More robust background processing

### Future Improvements

* [ ] Speaker diarization
* [ ] Better timestamp handling
* [ ] Meeting history and persistence
* [ ] Authentication
* [ ] User-specific meeting libraries
* [ ] Improved RAG evaluation
* [ ] Background task processing for long meetings
* [ ] Production monitoring and logging

---

## Key Learning

This project was built to understand how a complete GenAI application works beyond simply calling an LLM.

The main learning areas were:

* Audio preprocessing
* Speech-to-text systems
* LLM integration
* Prompt-based information extraction
* LangChain and LCEL
* Embeddings and vector databases
* Retrieval-Augmented Generation
* Application architecture
* API integration
* Dependency management
* Debugging external services
* Git/GitHub workflows
* Cloud deployment

The most important part of the project was understanding how these individual components fit together into one end-to-end system.

---

## Future Architecture

The current Streamlit application is being evolved toward a separated frontend/backend architecture:

```text
                    ┌─────────────────────┐
                    │   HTML/CSS/JS       │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │
                              HTTP
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Pipeline       │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
        Whisper             Gemini              RAG
        /Sarvam              LLM             + ChromaDB
```

This separation will make the application easier to deploy, maintain, and extend.

---

## Author

**Krittika Biswas**

BTech Computer Science & Engineering — AI/ML

GitHub:
https://github.com/richebiswas

---

## License

This project is intended primarily as a learning and portfolio project.
