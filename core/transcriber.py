
import whisper
import os
from pydub import AudioSegment
from sarvamai import SarvamAI


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v4")

SARVAM_PIECE_SECONDS = 25

_model = None


def load_model():

    global _model

    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"]


def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""

    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    client = SarvamAI(
        api_subscription_key=SARVAM_API_KEY
    )

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"

        piece.export(
            piece_path,
            format="wav"
        )

        try:

            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces} ..."
            )

            with open(piece_path, "rb") as f:

                response = client.speech_to_text.transcribe(
                    file=f,
                    model=SARVAM_MODEL,
                    mode="translate"
                )

            full_text += response.transcript + " "

        finally:

            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    if language.lower() == "hinglish":

        return transcribe_chunk_sarvam(
            chunk_path
        )

    return transcribe_chunk_whisper(
        chunk_path
    )


def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript = ""

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    print("Transcription completed.")

    return full_transcript.strip()

