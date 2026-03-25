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
    show_meter: bool = True,
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
                if show_meter:
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
    """Check TTS backend availability (lazy, cached after first call).

    Kokoro runs in a subprocess to isolate Metal GPU state, so we don't
    load the model here — just check if the import works.
    """
    global _tts_backend
    if _tts_backend is not None:
        return

    # Check if Kokoro/mlx-audio is importable
    try:
        import mlx_audio  # noqa: F401
        _tts_backend = "kokoro"
        log.info("Kokoro TTS available (will run in subprocess)")
    except ImportError:
        log.info("mlx-audio not available, using edge-tts fallback")
        _tts_backend = "edge-tts"


def _ensure_tts():
    """Check TTS backend availability."""
    _get_tts_model()


def _clean_for_tts(text: str) -> str:
    """Strip markdown and emoji from text for TTS."""
    import re as _re
    clean = text.replace("**", "").replace("*", "").replace("`", "")
    clean = _re.sub(
        r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FEFF\U0001FA00-\U0001FAFF]+',
        '', clean
    ).strip()
    if len(clean) > 500:
        clean = clean[:500] + "..."
    return clean


def speak(text: str) -> None:
    """Speak text aloud. Non-blocking.

    Queues text onto a background speech thread. Multiple calls are spoken
    in order — new text waits for previous speech to finish (no cancellation).
    Use stop_speech() to cancel everything.
    """
    clean = _clean_for_tts(text)
    if not clean:
        return
    _speech_queue.put(clean)
    _ensure_speech_worker()


def _ensure_speech_worker():
    """Start the background speech worker if not already running."""
    import threading
    global _speech_worker
    if _speech_worker is not None and _speech_worker.is_alive():
        return
    _speech_worker = threading.Thread(target=_speech_worker_loop, daemon=True)
    _speech_worker.start()


def _speech_worker_loop():
    """Background thread: drain the speech queue, speak each item in order."""
    import queue
    while True:
        try:
            text = _speech_queue.get(timeout=5)
        except queue.Empty:
            return  # idle for 5s — exit thread (will restart on next speak())
        try:
            _speak_one(text)
        except Exception as e:
            log.warning(f"TTS failed: {e}")
        finally:
            _speech_queue.task_done()


def _speak_one(text: str) -> None:
    """Speak a single piece of text synchronously (blocks until audio finishes)."""
    # Try Kokoro first (local, subprocess-isolated)
    if _tts_backend == "kokoro":
        try:
            _speak_kokoro(text, 0)
            return
        except Exception as e:
            log.warning(f"Kokoro TTS failed: {e}")

    # Fallback: Edge TTS (cloud)
    try:
        _speak_edge_tts(text, 0)
        return
    except Exception as e:
        log.warning(f"Edge TTS failed: {e}")

    # Last resort: macOS say
    _speak_say(text)


def _speak_kokoro(text: str, _epoch: int = 0) -> None:
    """Generate and play speech using Kokoro TTS in a subprocess.

    Runs in a separate process to isolate Metal GPU operations from the
    main process (prevents segfaults when mlx-whisper and Kokoro compete
    for GPU memory).
    """
    import tempfile

    config = _get_stt_config()
    voice = config.get("kokoro_voice", KOKORO_VOICE)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        txt_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-c", _KOKORO_SUBPROCESS_SCRIPT,
             txt_path, wav_path, voice],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode()[:200] if result.stderr else ""
            raise RuntimeError(f"Kokoro subprocess failed (rc={result.returncode}): {stderr}")

        subprocess.run(
            ["afplay", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        Path(txt_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)


# Script run in a subprocess for Kokoro TTS generation.
# Isolates Metal GPU operations from the main process.
_KOKORO_SUBPROCESS_SCRIPT = """\
import sys, contextlib, io
txt_path, wav_path, voice = sys.argv[1], sys.argv[2], sys.argv[3]

text = open(txt_path).read()

with contextlib.redirect_stdout(io.StringIO()):
    from mlx_audio.tts.utils import load_model
    model = load_model("mlx-community/Kokoro-82M-bf16")
    import numpy as np, soundfile as sf
    chunks = []
    for result in model.generate(text, voice=voice):
        chunks.append(result.audio)
    if chunks:
        sf.write(wav_path, np.concatenate(chunks), 24000)
"""


def _speak_edge_tts(text: str, _epoch: int = 0) -> None:
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


# Speech queue — speak() enqueues, worker thread drains in order
import queue as _queue_mod
_speech_queue: _queue_mod.Queue = _queue_mod.Queue()
_speech_worker = None




def wait_for_speech() -> None:
    """Wait for all queued TTS to finish playing."""
    _speech_queue.join()  # blocks until queue is drained
    # Also wait for worker thread to finish current playback
    if _speech_worker is not None and _speech_worker.is_alive():
        _speech_worker.join(timeout=15)


def stop_speech() -> None:
    """Kill any running TTS playback and clear the queue."""
    # Clear pending items
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
            _speech_queue.task_done()
        except _queue_mod.Empty:
            break
    # Kill currently playing audio
    try:
        subprocess.run(["pkill", "-f", "afplay"], capture_output=True)
    except Exception:
        pass


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
    audio = record_until_silence(show_meter=False)
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
