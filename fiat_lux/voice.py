"""Voice input/output — microphone capture, Whisper transcription, and TTS.

Requires optional dependencies: `uv sync --extra voice`
  - openai-whisper (local speech-to-text)
  - sounddevice (microphone capture)

TTS uses macOS `say` command — no extra deps needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
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

# Set by agent.py to enable volume meter display
_console = None

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

            # Show volume meter if console is available
            if _console is not None:
                bar = format_volume_bar(level)
                status = "[lux.highlight]recording[/lux.highlight]" if has_speech else "[lux.dim]waiting[/lux.dim]"
                _console.print(f"\r  {bar} {status}", end="")

            if level > threshold:
                has_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1

            # Stop after enough silence, but only if we heard speech first
            if has_speech and silence_chunks >= silence_limit:
                break

    # Clear the volume meter line
    if _console is not None:
        _console.print("\r" + " " * 60 + "\r", end="")

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


# ---------------------------------------------------------------------------
# Text-to-speech via macOS `say`
# ---------------------------------------------------------------------------

TTS_VOICE = "Samantha"  # high-quality macOS voice


def speak(text: str, voice: str = TTS_VOICE) -> None:
    """Speak text aloud using macOS TTS. Non-blocking."""
    if sys.platform != "darwin":
        return
    # Strip markdown-style formatting that sounds weird spoken
    clean = text.replace("**", "").replace("*", "").replace("`", "")
    # Limit length — don't read back huge responses
    if len(clean) > 500:
        clean = clean[:500] + "..."
    try:
        subprocess.Popen(
            ["say", "-v", voice, clean],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # `say` not available


# ---------------------------------------------------------------------------
# Volume meter for recording feedback
# ---------------------------------------------------------------------------

def format_volume_bar(rms_level: float, width: int = 20) -> str:
    """Format a volume level as a visual bar. Returns a string like '|||||     '."""
    filled = min(width, int(rms_level * width * 10))  # scale up for visibility
    return "\u2588" * filled + "\u2591" * (width - filled)
