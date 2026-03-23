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
SILENCE_DURATION = 2.0  # seconds of silence after speech to auto-stop
MAX_DURATION = 120  # max recording seconds (silence detection is the real stop)
CALIBRATION_SECONDS = 0.5  # measure ambient noise before listening
THRESHOLD_MULTIPLIER = 3.0  # speech must be Nx louder than ambient

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
) -> np.ndarray | None:
    """Record from microphone until silence is detected.

    Auto-calibrates the noise threshold from the first 0.5s of ambient audio.
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

    from rich.live import Live
    from rich.text import Text

    live_ctx = Live("", console=_console, refresh_per_second=10) if _console else None
    if live_ctx:
        live_ctx.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            # Calibrate: measure ambient noise for 0.5s
            cal_chunks = int(CALIBRATION_SECONDS / 0.1)
            ambient_levels = []
            for _ in range(cal_chunks):
                audio, _ = stream.read(chunk_size)
                ambient_levels.append(_rms(audio))
                if live_ctx:
                    live_ctx.update(Text("  calibrating..."))

            ambient_rms = max(ambient_levels) if ambient_levels else 0.005
            threshold = ambient_rms * THRESHOLD_MULTIPLIER

            # Record until silence after speech
            for _ in range(max_chunks):
                audio, _ = stream.read(chunk_size)
                chunks.append(audio.copy())

                level = _rms(audio)

                # Update volume meter
                if live_ctx is not None:
                    bar = format_volume_bar(level)
                    if has_speech:
                        status = "recording"
                    elif level > threshold:
                        status = "hearing you"
                    else:
                        status = "waiting"
                    live_ctx.update(Text(f"  {bar} {status}"))

                if level > threshold:
                    has_speech = True
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                if has_speech and silence_chunks >= silence_limit:
                    break
    except KeyboardInterrupt:
        # Ctrl+C during recording — return what we have or None
        pass
    finally:
        if live_ctx:
            live_ctx.stop()

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

    try:
        audio = record_until_silence()
    except KeyboardInterrupt:
        return None

    if audio is None:
        return None

    # Show spinner during transcription
    if _console is not None:
        with _console.status("[lux.highlight]Transcribing...", spinner="dots"):
            return transcribe(audio)
    return transcribe(audio)


# ---------------------------------------------------------------------------
# Text-to-speech via macOS `say`
# ---------------------------------------------------------------------------

TTS_VOICE = "en-US-EmmaNeural"


def speak(text: str, voice: str = TTS_VOICE) -> None:
    """Speak text aloud using Edge TTS (neural voice). Non-blocking."""
    import threading

    # Strip markdown formatting that sounds weird spoken
    clean = text.replace("**", "").replace("*", "").replace("`", "")
    # Limit length
    if len(clean) > 500:
        clean = clean[:500] + "..."

    def _speak():
        import asyncio
        import tempfile
        try:
            import edge_tts

            async def _generate_and_play():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp = f.name
                comm = edge_tts.Communicate(clean, voice)
                await comm.save(tmp)
                # afplay plays audio directly through speakers, no GUI
                subprocess.run(
                    ["afplay", tmp],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            asyncio.run(_generate_and_play())
        except Exception:
            # Fall back to macOS say
            try:
                subprocess.run(
                    ["say", clean],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

    # Run in background thread so it doesn't block the UI
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    _speak_threads.append(t)


# Track active speak threads so we can wait for them before exit
_speak_threads: list[threading.Thread] = []


def wait_for_speech() -> None:
    """Wait for all pending TTS to finish playing. Call before exiting."""
    for t in _speak_threads:
        t.join(timeout=30)
    _speak_threads.clear()


# ---------------------------------------------------------------------------
# Volume meter for recording feedback
# ---------------------------------------------------------------------------

def format_volume_bar(rms_level: float, width: int = 20) -> str:
    """Format a volume level as a visual bar. Returns a string like '|||||     '."""
    filled = min(width, int(rms_level * width * 10))  # scale up for visibility
    return "\u2588" * filled + "\u2591" * (width - filled)
