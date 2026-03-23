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
    Uses a callback-based stream so Ctrl+C works reliably.
    Returns a numpy array of float32 audio at 16kHz, or None if no speech detected.
    """
    import numpy as np
    import sounddevice as sd
    import queue
    import time as _time

    from rich.live import Live
    from rich.text import Text

    # Audio queue — callback pushes chunks, main thread reads them
    audio_queue: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    chunks = []
    silence_chunks = 0
    silence_limit = int(silence_seconds / 0.1)
    has_speech = False

    live_ctx = Live("", console=_console, refresh_per_second=10) if _console else None
    if live_ctx:
        live_ctx.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=_callback,
        ):
            # Calibrate: measure ambient noise for 0.5s
            ambient_levels = []
            cal_end = _time.monotonic() + CALIBRATION_SECONDS
            while _time.monotonic() < cal_end:
                try:
                    audio = audio_queue.get(timeout=0.2)
                    ambient_levels.append(_rms(audio))
                    if live_ctx:
                        live_ctx.update(Text("  calibrating..."))
                except queue.Empty:
                    pass

            ambient_rms = max(ambient_levels) if ambient_levels else 0.005
            threshold = ambient_rms * THRESHOLD_MULTIPLIER

            # Record until silence after speech
            deadline = _time.monotonic() + max_seconds
            while _time.monotonic() < deadline:
                try:
                    audio = audio_queue.get(timeout=0.15)
                except queue.Empty:
                    continue

                chunks.append(audio)
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
        t.join(timeout=15)
    _speak_threads.clear()


def stop_speech() -> None:
    """Kill any running TTS playback immediately."""
    import signal as _signal
    # Kill any afplay processes
    try:
        subprocess.run(["pkill", "-f", "afplay"], capture_output=True)
    except Exception:
        pass
    _speak_threads.clear()


# ---------------------------------------------------------------------------
# Volume meter for recording feedback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

WAKE_PHRASES = {"hey lux", "hey lucks", "hey lucks", "a lux", "hey luck", "haylux", "hey, lux"}
WAKE_LISTEN_SECONDS = 2.5  # how long each wake word check records


def detect_wake_word() -> bool:
    """Record a short clip and check if it contains 'Hey Lux'.

    Returns True if wake word detected, False otherwise.
    Designed to be called in a tight loop for always-on listening.
    """
    import numpy as np
    import sounddevice as sd
    import queue

    audio_queue: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    chunks = []
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=_callback,
        ):
            import time as _time
            deadline = _time.monotonic() + WAKE_LISTEN_SECONDS
            while _time.monotonic() < deadline:
                try:
                    audio = audio_queue.get(timeout=0.15)
                    chunks.append(audio)
                except queue.Empty:
                    continue
    except KeyboardInterrupt:
        raise

    if not chunks:
        return False

    audio = np.concatenate(chunks).flatten()

    # Only transcribe if there's some audio energy (skip pure silence)
    if _rms(audio) < 0.005:
        return False

    # Quick transcribe — check for wake word
    model = _get_whisper_model()
    result = model.transcribe(audio, language="en", fp16=False)
    text = result["text"].strip().lower()

    # Check for wake word variants
    for phrase in WAKE_PHRASES:
        if phrase in text:
            return True
    return False


def format_volume_bar(rms_level: float, width: int = 20) -> str:
    """Format a volume level as a visual bar. Returns a string like '|||||     '."""
    filled = min(width, int(rms_level * width * 10))  # scale up for visibility
    return "\u2588" * filled + "\u2591" * (width - filled)
