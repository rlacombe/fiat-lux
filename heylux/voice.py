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
SILENCE_DURATION = 1.0  # seconds of silence after speech to auto-stop
MAX_DURATION = 10  # max recording seconds (short voice commands)
CALIBRATION_SECONDS = 0.3  # measure ambient noise before listening
THRESHOLD_MULTIPLIER = 2.5  # speech must be Nx louder than ambient
THRESHOLD_FLOOR = 0.005  # minimum absolute threshold (prevents near-zero ambient issues)
MIN_RECORD_SECONDS = 0.5  # always record at least this long before checking silence

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
        model_name = config.get("model", "mlx-community/whisper-tiny")
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

    Auto-calibrates the noise threshold from the first 0.3s of ambient audio.
    Uses a callback-based stream so Ctrl+C works reliably.
    Returns a numpy array of float32 audio at 16kHz, or None if no speech detected.
    """
    import numpy as np
    import sounddevice as sd
    import queue
    import time as _time

    t_start = _time.monotonic()
    log.info("[record] starting mic capture")

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
            # Calibrate: measure ambient noise
            ambient_levels = []
            cal_end = _time.monotonic() + CALIBRATION_SECONDS
            while _time.monotonic() < cal_end:
                try:
                    audio = audio_queue.get(timeout=0.2)
                    ambient_levels.append(_rms(audio))
                except queue.Empty:
                    pass

            ambient_rms = max(ambient_levels) if ambient_levels else 0.005
            threshold = max(ambient_rms * THRESHOLD_MULTIPLIER, THRESHOLD_FLOOR)
            t_calibrated = _time.monotonic()
            log.info(f"[record] calibrated in {t_calibrated - t_start:.2f}s "
                     f"(ambient={ambient_rms:.4f}, threshold={threshold:.4f})")

            # Record until silence after speech
            deadline = _time.monotonic() + max_seconds
            min_end = _time.monotonic() + MIN_RECORD_SECONDS
            t_speech_start = None
            while _time.monotonic() < deadline:
                try:
                    audio = audio_queue.get(timeout=0.15)
                except queue.Empty:
                    continue

                chunks.append(audio)
                level = _rms(audio)

                if show_meter:
                    bar = format_volume_bar(level)
                    if has_speech:
                        _status(f"{bar} recording")
                    elif level > threshold:
                        _status(f"{bar} hearing you")
                    else:
                        _status(f"{bar} waiting")

                if level > threshold:
                    if not has_speech:
                        t_speech_start = _time.monotonic()
                        log.info(f"[record] speech detected at {t_speech_start - t_start:.2f}s "
                                 f"(level={level:.4f} > threshold={threshold:.4f})")
                    has_speech = True
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                if has_speech and silence_chunks >= silence_limit and _time.monotonic() > min_end:
                    t_silence = _time.monotonic()
                    log.info(f"[record] silence detected at {t_silence - t_start:.2f}s "
                             f"(speech duration ~{t_silence - t_speech_start - silence_seconds:.1f}s + "
                             f"{silence_seconds}s silence)")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if _console is not None:
            _console.file.write("\r\033[K")
            _console.file.flush()

    t_end = _time.monotonic()
    if not has_speech:
        log.info(f"[record] no speech detected ({t_end - t_start:.2f}s)")
        return None

    import numpy as np
    audio_out = np.concatenate(chunks).flatten()
    audio_secs = len(audio_out) / SAMPLE_RATE
    log.info(f"[record] done: {audio_secs:.1f}s audio captured in {t_end - t_start:.2f}s total")
    return audio_out


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
    """Check TTS backend availability and pre-start worker."""
    _get_tts_model()
    warm_kokoro_worker()


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
    import time as _time
    t0 = _time.monotonic()

    # Try Kokoro first (local, subprocess-isolated)
    if _tts_backend == "kokoro":
        try:
            _speak_kokoro(text)
            log.info(f"[timing] tts_kokoro={_time.monotonic() - t0:.2f}s for '{text[:40]}'")
            return
        except Exception as e:
            log.warning(f"Kokoro TTS failed: {e}")

    # Fallback: Edge TTS (cloud)
    try:
        _speak_edge_tts(text)
        log.info(f"[timing] tts_edge={_time.monotonic() - t0:.2f}s")
        return
    except Exception as e:
        log.warning(f"Edge TTS failed: {e}")

    # Last resort: macOS say
    _speak_say(text)
    log.info(f"[timing] tts_say={_time.monotonic() - t0:.2f}s")


def _speak_kokoro(text: str) -> None:
    """Generate and play speech using persistent Kokoro TTS worker subprocess.

    The worker process stays alive between calls — model is loaded once,
    subsequent generations are fast (~200ms).
    """
    import tempfile
    import time as _time

    t0 = _time.monotonic()
    config = _get_stt_config()
    voice = config.get("kokoro_voice", KOKORO_VOICE)

    worker = _get_kokoro_worker()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        # Send request to persistent worker
        request = json.dumps({"text": text, "voice": voice, "wav_path": wav_path}) + "\n"
        worker.stdin.write(request.encode())
        worker.stdin.flush()

        # Wait for response
        response_line = worker.stdout.readline()
        if not response_line:
            raise RuntimeError("Kokoro worker died")

        response = json.loads(response_line.decode())
        if response.get("error"):
            raise RuntimeError(f"Kokoro worker: {response['error']}")

        t_generated = _time.monotonic()
        log.info(f"[tts] kokoro generated in {t_generated - t0:.2f}s: '{text[:50]}'")

        subprocess.run(
            ["afplay", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"[tts] playback finished in {_time.monotonic() - t_generated:.2f}s")
    except (BrokenPipeError, OSError):
        # Worker died — reset and retry once
        _kill_kokoro_worker()
        raise RuntimeError("Kokoro worker crashed, will restart on next call")
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Persistent Kokoro worker subprocess
# ---------------------------------------------------------------------------

_kokoro_worker = None


def _get_kokoro_worker():
    """Get or start the persistent Kokoro TTS worker subprocess."""
    global _kokoro_worker
    if _kokoro_worker is not None and _kokoro_worker.poll() is None:
        return _kokoro_worker

    log.info("Starting persistent Kokoro TTS worker...")
    _kokoro_worker = subprocess.Popen(
        [sys.executable, "-c", _KOKORO_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    # Wait for "ready" signal
    ready = _kokoro_worker.stdout.readline()
    if not ready or b"ready" not in ready:
        _kokoro_worker.kill()
        _kokoro_worker = None
        raise RuntimeError("Kokoro worker failed to start")
    log.info("Kokoro TTS worker ready")
    return _kokoro_worker


def _kill_kokoro_worker():
    """Kill the persistent Kokoro worker."""
    global _kokoro_worker
    if _kokoro_worker is not None:
        try:
            _kokoro_worker.kill()
        except Exception:
            pass
        _kokoro_worker = None


def warm_kokoro_worker():
    """Pre-start the Kokoro worker and warm the pipeline.

    Call during startup alongside ensure_model(). The first generation
    triggers pipeline compilation (~1.5s), so we do a dummy generation
    here so the first real TTS call is instant (~0.1s).
    """
    if _tts_backend != "kokoro":
        return
    try:
        import tempfile
        worker = _get_kokoro_worker()
        config = _get_stt_config()
        voice = config.get("kokoro_voice", KOKORO_VOICE)
        # Dummy generation to warm the pipeline (first call compiles it)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        request = json.dumps({"text": ".", "voice": voice, "wav_path": wav_path}) + "\n"
        worker.stdin.write(request.encode())
        worker.stdin.flush()
        worker.stdout.readline()  # wait for completion
        Path(wav_path).unlink(missing_ok=True)
        log.info("Kokoro pipeline warmed up")
    except Exception as e:
        log.warning(f"Failed to warm Kokoro worker: {e}")


# Worker script: loads model once, then loops reading JSON requests from stdin.
_KOKORO_WORKER_SCRIPT = """\
import sys, json, contextlib, io

# Load model once at startup
with contextlib.redirect_stdout(io.StringIO()):
    from mlx_audio.tts.utils import load_model
    import numpy as np, soundfile as sf
    model = load_model("mlx-community/Kokoro-82M-bf16")

# Signal ready
sys.stdout.buffer.write(b'{"ready": true}\\n')
sys.stdout.buffer.flush()

# Process requests
for line in sys.stdin.buffer:
    try:
        req = json.loads(line.decode())
        text = req["text"]
        voice = req["voice"]
        wav_path = req["wav_path"]

        with contextlib.redirect_stdout(io.StringIO()):
            chunks = []
            for result in model.generate(text, voice=voice):
                chunks.append(result.audio)
            if chunks:
                sf.write(wav_path, np.concatenate(chunks), 24000)

        sys.stdout.buffer.write(b'{"ok": true}\\n')
        sys.stdout.buffer.flush()
    except Exception as e:
        sys.stdout.buffer.write(json.dumps({"error": str(e)}).encode() + b'\\n')
        sys.stdout.buffer.flush()
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
    import time as _time
    t0 = _time.monotonic()

    audio = record_until_silence(show_meter=False)
    if audio is None:
        return None

    t_recorded = _time.monotonic()
    audio_secs = len(audio) / SAMPLE_RATE
    log.info(f"[timing] recorded {audio_secs:.1f}s audio in {t_recorded - t0:.2f}s")

    # Transcribe
    if _console is not None:
        with _console.status("[lux.highlight]Transcribing...", spinner="dots"):
            text = transcribe(audio)
    else:
        text = transcribe(audio)

    t_transcribed = _time.monotonic()
    log.info(f"[timing] transcribed in {t_transcribed - t_recorded:.2f}s")

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
