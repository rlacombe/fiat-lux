"""Voice input/output — microphone capture, STT transcription, and TTS.

Requires optional dependencies: `uv sync --extra voice`
  - lightning-whisper-mlx (fast local STT on Apple Silicon)
  - sounddevice (microphone capture)
  - mlx-audio (Kokoro TTS — fast local text-to-speech)

Fallback chain:
  STT: lightning-whisper-mlx → openai-whisper → error
  TTS: Kokoro (mlx-audio) → edge-tts → macOS `say`
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("heylux.voice")

CONFIG_DIR = Path.home() / ".config" / "heylux"
VOICE_CONFIG = CONFIG_DIR / "voice.json"

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1  # mono
SILENCE_DURATION = 1.0  # seconds of silence after speech to auto-stop (was 2.0)
MAX_DURATION = 120  # max recording seconds (silence detection is the real stop)
CALIBRATION_SECONDS = 0.3  # measure ambient noise before listening (was 0.5)
THRESHOLD_MULTIPLIER = 3.5  # speech must be Nx louder than ambient (rejects keyboard clicks)
MIN_RECORD_SECONDS = 0.5  # always record at least this long before checking silence (was 1.0)

# Set by agent.py to enable volume meter display
_console = None

# Lazy-loaded STT model
_model = None
_stt_backend: str | None = None  # "lightning-mlx" or "openai-whisper"


# ---------------------------------------------------------------------------
# STT — Speech-to-Text
# ---------------------------------------------------------------------------

def _get_stt_config() -> dict[str, str]:
    """Get STT configuration from voice.json."""
    if VOICE_CONFIG.exists():
        try:
            return json.loads(VOICE_CONFIG.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _get_whisper_model():
    """Load STT model (lazy, cached after first call).

    Tries mlx-whisper first (5-10x faster on Apple Silicon via Metal GPU),
    falls back to openai-whisper.
    """
    global _model, _stt_backend
    if _model is not None:
        return _model

    config = _get_stt_config()

    # Try mlx-whisper first (fast, Apple Silicon native via Metal)
    try:
        import mlx_whisper
        import numpy as _np
        model_name = config.get("model", "mlx-community/distil-whisper-large-v3")
        log.info(f"Loading mlx-whisper model: {model_name}")
        # Warm up: run a tiny transcription to force model download + compilation.
        # Without this, the first real transcription is slow (~5s extra).
        _silence = _np.zeros(SAMPLE_RATE, dtype=_np.float32)  # 1s of silence
        mlx_whisper.transcribe(_silence, path_or_hf_repo=model_name, language="en")
        _model = model_name
        _stt_backend = "mlx-whisper"
        log.info("mlx-whisper loaded and warmed up")
        return _model
    except ImportError:
        log.info("mlx-whisper not available, trying openai-whisper")
    except Exception as e:
        log.warning(f"mlx-whisper failed: {e}, trying openai-whisper")

    # Fallback: openai-whisper
    try:
        import whisper
        model_name = config.get("model", "base")
        log.info(f"Loading openai-whisper model: {model_name}")
        _model = whisper.load_model(model_name)
        _stt_backend = "openai-whisper"
        log.info("openai-whisper loaded successfully")
        return _model
    except ImportError:
        raise ImportError(
            "No STT backend available. Install: uv sync --extra voice"
        )


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

    # Audio queue — callback pushes chunks, main thread reads them
    audio_queue: queue.Queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    chunks = []
    silence_chunks = 0
    silence_limit = int(silence_seconds / 0.1)
    has_speech = False

    def _status(text: str) -> None:
        """Overwrite the status line in place."""
        if _console is not None:
            # Use ANSI escape to clear line and write
            _console.file.write(f"\r\033[K  {text}")
            _console.file.flush()

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
                except queue.Empty:
                    pass

            ambient_rms = max(ambient_levels) if ambient_levels else 0.005
            threshold = ambient_rms * THRESHOLD_MULTIPLIER

            # Record until silence after speech
            deadline = _time.monotonic() + max_seconds
            min_end = _time.monotonic() + MIN_RECORD_SECONDS
            while _time.monotonic() < deadline:
                try:
                    audio = audio_queue.get(timeout=0.15)
                except queue.Empty:
                    continue

                chunks.append(audio)
                level = _rms(audio)

                # Update volume meter — single line, overwritten
                bar = format_volume_bar(level)
                if has_speech:
                    _status(f"{bar} recording")
                elif level > threshold:
                    _status(f"{bar} hearing you")
                else:
                    _status(f"{bar} waiting")

                if level > threshold:
                    has_speech = True
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                if has_speech and silence_chunks >= silence_limit and _time.monotonic() > min_end:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        # Clear the status line
        if _console is not None:
            _console.file.write("\r\033[K")
            _console.file.flush()

    if not has_speech:
        return None

    import numpy as np
    return np.concatenate(chunks).flatten()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using the loaded STT model.

    Automatically uses whichever backend was loaded (mlx-whisper or openai-whisper).
    Logs timing for performance monitoring.

    Args:
        audio: float32 numpy array at 16kHz.

    Returns:
        Transcribed text, stripped.
    """
    import time as _time
    model = _get_whisper_model()

    t0 = _time.monotonic()
    audio_secs = len(audio) / SAMPLE_RATE

    if _stt_backend == "mlx-whisper":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model,
            language="en",
        )
        text = result.get("text", "").strip()
    else:
        result = model.transcribe(
            audio,
            language="en",
            fp16=False,
        )
        text = result["text"].strip()

    elapsed = _time.monotonic() - t0
    log.info(f"Transcribed {audio_secs:.1f}s audio in {elapsed:.2f}s: '{text[:60]}'")
    return text


def ensure_model() -> None:
    """Pre-load the STT model, downloading if needed.

    Call this before the first listen to avoid download during recording.
    """
    _get_whisper_model()


def listen_once() -> str | None:
    """Record from mic and transcribe. Returns text or None if no speech.

    Raises ImportError if voice dependencies aren't installed.
    """
    try:
        import sounddevice  # noqa: F401
        _get_whisper_model()  # verify STT backend is available
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
# TTS — Text-to-Speech
#
# Priority: Kokoro (mlx-audio, local) → edge-tts (cloud) → macOS say
# ---------------------------------------------------------------------------

# Kokoro voice preset (see mlx-audio docs for available voices)
KOKORO_VOICE = "af_aoede"
# Edge TTS fallback voice
EDGE_TTS_VOICE = "en-US-AriaNeural"

# Lazy-loaded Kokoro TTS model
_tts_model = None
_tts_backend: str | None = None  # "kokoro", "edge-tts", "say"


def _get_tts_model():
    """Load Kokoro TTS model (lazy, cached after first call)."""
    global _tts_model, _tts_backend
    if _tts_model is not None:
        return _tts_model

    # Try Kokoro via mlx-audio (fast, local, no network)
    try:
        import io as _io, contextlib as _ctx
        from mlx_audio.tts.utils import load_model
        log.info("Loading Kokoro TTS model...")
        # Suppress noisy "Creating new KokoroPipeline" print from library
        with _ctx.redirect_stdout(_io.StringIO()):
            _tts_model = load_model("mlx-community/Kokoro-82M-bf16")
        _tts_backend = "kokoro"
        log.info("Kokoro TTS loaded successfully")
        return _tts_model
    except ImportError:
        log.info("mlx-audio not available for TTS")
    except Exception as e:
        log.warning(f"Kokoro TTS failed to load: {e}")

    # Mark as unavailable — speak() will use edge-tts or say fallback
    _tts_backend = "edge-tts"
    return None


def _ensure_tts():
    """Pre-load TTS model if available."""
    _get_tts_model()


def speak(text: str) -> None:
    """Speak text aloud. Non-blocking.

    Uses Kokoro (local, fast) if available, falls back to Edge TTS (cloud),
    then macOS `say`. Each call cancels any in-flight speech using an epoch
    counter to prevent overlapping audio.
    """
    import threading
    global _speak_epoch

    # Strip markdown formatting that sounds weird spoken
    clean = text.replace("**", "").replace("*", "").replace("`", "")
    # Limit length
    if len(clean) > 500:
        clean = clean[:500] + "..."

    # Bump epoch so any in-flight speech thread knows to bail out
    _speak_epoch += 1
    my_epoch = _speak_epoch

    # Kill any currently playing audio
    _stop_current_speech()

    def _speak():
        if _speak_epoch != my_epoch:
            return

        # Try Kokoro first (local, fast)
        if _tts_backend == "kokoro" and _tts_model is not None:
            try:
                _speak_kokoro(clean, my_epoch)
                return
            except Exception as e:
                log.warning(f"Kokoro TTS failed: {e}")

        # Fallback: Edge TTS (cloud)
        try:
            _speak_edge_tts(clean, my_epoch)
            return
        except Exception as e:
            log.warning(f"Edge TTS failed: {e}")

        # Last resort: macOS say
        if _speak_epoch != my_epoch:
            return
        _speak_say(clean)

    # Run in background thread so it doesn't block the UI
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    _speak_threads.append(t)


def _speak_kokoro(text: str, epoch: int) -> None:
    """Generate and play speech using Kokoro TTS (local, ~200ms)."""
    import numpy as np
    import tempfile
    import soundfile as sf

    config = _get_stt_config()
    voice = config.get("kokoro_voice", KOKORO_VOICE)

    # Generate audio — Kokoro yields chunks
    # Suppress noisy "Creating new KokoroPipeline" print from library
    import io as _io, contextlib as _ctx
    audio_chunks = []
    with _ctx.redirect_stdout(_io.StringIO()):
        for result in _tts_model.generate(text, voice=voice):
            if _speak_epoch != epoch:
                return
            audio_chunks.append(result.audio)

    if _speak_epoch != epoch:
        return

    if not audio_chunks:
        return

    # Concatenate and save to temp file for playback
    audio = np.concatenate(audio_chunks)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, 24000)  # Kokoro outputs at 24kHz
        if _speak_epoch != epoch:
            return
        subprocess.run(
            ["afplay", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _speak_edge_tts(text: str, epoch: int) -> None:
    """Generate and play speech using Edge TTS (cloud, ~1-2s)."""
    import asyncio
    import tempfile

    import edge_tts

    config = _get_stt_config()
    voice = config.get("edge_voice", EDGE_TTS_VOICE)

    async def _generate_and_play():
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        comm = edge_tts.Communicate(text, voice)
        await comm.save(tmp)
        if _speak_epoch != epoch:
            Path(tmp).unlink(missing_ok=True)
            return
        try:
            subprocess.run(
                ["afplay", tmp],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

    asyncio.run(_generate_and_play())


def _speak_say(text: str) -> None:
    """Speak using macOS `say` command (instant, lower quality)."""
    try:
        subprocess.run(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


# Track active speak threads so we can wait for them before exit
_speak_threads: list = []
_speak_epoch: int = 0


def _stop_current_speech() -> None:
    """Kill any currently playing afplay so we don't overlap."""
    try:
        subprocess.run(["pkill", "-f", "afplay"], capture_output=True)
    except Exception:
        pass
    # Clean up finished threads
    _speak_threads[:] = [t for t in _speak_threads if t.is_alive()]


def wait_for_speech() -> None:
    """Wait for all pending TTS to finish playing. Call before exiting."""
    for t in _speak_threads:
        t.join(timeout=15)
    _speak_threads.clear()


def stop_speech() -> None:
    """Kill any running TTS playback immediately."""
    try:
        subprocess.run(["pkill", "-f", "afplay"], capture_output=True)
    except Exception:
        pass
    _speak_threads.clear()


# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

WAKE_PHRASES = {
    "hey lux", "hey lucks", "hey luck", "hey lox", "hey locks",
    "hey docs", "hey docks", "hey vox", "hey box",
    "hey lax", "hey luxe", "hey luks", "hey luke",
    "a lux", "haylux", "hey, lux", "he lux", "hey lex",
    "hey, lex", "hey, lucks", "hey, luck", "hey, lox",
}


def listen_for_wake_command() -> str | None:
    """Continuously listen, and when speech starts, record it all.

    Transcribes the result. If it starts with 'Hey Lux', strips the
    wake word and returns the command. If no wake word, returns None.

    This captures "Hey Lux, turn my lights blue" in a single recording.
    """
    audio = record_until_silence()
    if audio is None:
        return None

    # Transcribe
    if _console is not None:
        with _console.status("[lux.highlight]Transcribing...", spinner="dots"):
            text = transcribe(audio)
    else:
        text = transcribe(audio)

    if not text:
        return None

    text_lower = text.lower().strip()

    # Strip common STT artifacts at the start (filler words, punctuation)
    for prefix in ("hi. ", "hi, ", "hello. ", "hello, ", "oh, ", "um, ", "uh, "):
        if text_lower.startswith(prefix):
            text_lower = text_lower[len(prefix):]
            text = text[len(prefix):]
            break

    # Log what STT heard for debugging
    log.info(f"STT heard: '{text_lower}'")

    # Check if it starts with a wake phrase
    for phrase in WAKE_PHRASES:
        if text_lower.startswith(phrase):
            # Strip the wake word and return the command
            command = text[len(phrase):].strip().lstrip(".,!?:").strip()
            if command:
                return command
            # They just said "Hey Lux" with no command — return empty
            # so the caller knows to prompt for more
            return ""

    return None


def format_volume_bar(rms_level: float, width: int = 20) -> str:
    """Format a volume level as a visual bar. Returns a string like '|||||     '."""
    filled = min(width, int(rms_level * width * 10))  # scale up for visibility
    return "\u2588" * filled + "\u2591" * (width - filled)
