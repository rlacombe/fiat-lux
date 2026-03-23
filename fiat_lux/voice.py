"""Voice input — microphone capture and local Whisper transcription.

Requires optional dependencies: `uv sync --extra voice`
  - openai-whisper (local speech-to-text)
  - sounddevice (microphone capture)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
VOICE_CONFIG = CONFIG_DIR / "voice.json"

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1  # mono
SILENCE_THRESHOLD = 0.01  # RMS below this = silence
SILENCE_DURATION = 1.5  # seconds of silence to auto-stop
MAX_DURATION = 15  # max recording seconds

# Lazy-loaded Whisper model
_model = None


def _get_model_name() -> str:
    """Get configured Whisper model name."""
    if VOICE_CONFIG.exists():
        try:
            config = json.loads(VOICE_CONFIG.read_text())
            return config.get("model", "base")
        except (json.JSONDecodeError, ValueError):
            pass
    return "base"


def _get_whisper_model():
    """Load Whisper model (lazy, cached after first call)."""
    global _model
    if _model is None:
        import whisper

        model_name = _get_model_name()
        _model = whisper.load_model(model_name)
    return _model


def _rms(audio: np.ndarray) -> float:
    """Compute root mean square of audio chunk."""
    import numpy as np
    return float(np.sqrt(np.mean(audio**2)))


def record_until_silence(
    max_seconds: float = MAX_DURATION,
    silence_seconds: float = SILENCE_DURATION,
    threshold: float = SILENCE_THRESHOLD,
) -> np.ndarray | None:
    """Record from microphone until silence is detected.

    Returns a numpy array of float32 audio at 16kHz, or None if no speech detected.
    """
    import numpy as np
    import sounddevice as sd

    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    chunks = []
    silence_chunks = 0
    silence_limit = int(silence_seconds / 0.1)
    max_chunks = int(max_seconds / 0.1)
    has_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        for _ in range(max_chunks):
            audio, _ = stream.read(chunk_size)
            chunks.append(audio.copy())

            level = _rms(audio)
            if level > threshold:
                has_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1

            # Stop after enough silence, but only if we heard speech first
            if has_speech and silence_chunks >= silence_limit:
                break

    if not has_speech:
        return None

    import numpy as np
    return np.concatenate(chunks).flatten()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using local Whisper model.

    Args:
        audio: float32 numpy array at 16kHz.

    Returns:
        Transcribed text, stripped.
    """
    model = _get_whisper_model()

    # Whisper expects float32 audio
    result = model.transcribe(
        audio,
        language="en",
        fp16=False,  # CPU/MPS compatibility
    )
    return result["text"].strip()


def ensure_model() -> None:
    """Pre-load the Whisper model, downloading if needed.

    Call this before the first listen to avoid download during recording.
    """
    _get_whisper_model()


def listen_once() -> str | None:
    """Record from mic and transcribe. Returns text or None if no speech.

    Raises ImportError if voice dependencies aren't installed.
    """
    try:
        import sounddevice  # noqa: F401
        import whisper  # noqa: F401
    except ImportError:
        raise ImportError(
            "Voice dependencies not installed. Run: uv sync --extra voice"
        )

    audio = record_until_silence()
    if audio is None:
        return None
    return transcribe(audio)
