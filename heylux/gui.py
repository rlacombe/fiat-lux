"""Hey Lux! — macOS menubar app for voice-controlled lighting.

Lives in the status bar, passively listens for "Hey Lux", then
records a voice command, executes it via the daemon, and responds
with TTS. Minimal CPU/memory when idle.

Install: uv sync --extra gui
Run: lux-app
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import rumps

# Lazy imports for voice — only loaded when needed
_voice_loaded = False
_whisper_model = None

# App state
_is_listening = False
_daemon_started = False

CONFIG_DIR = Path.home() / ".config" / "heylux"
SOCKET_PATH = CONFIG_DIR / "lux.sock"
PID_FILE = CONFIG_DIR / "lux.pid"


# ---------------------------------------------------------------------------
# Icons (simple Unicode for the menubar)
# ---------------------------------------------------------------------------

ICON_IDLE = "💡"
ICON_LISTENING = "🎙️"
ICON_PROCESSING = "⚡"
ICON_SPEAKING = "🔊"


# ---------------------------------------------------------------------------
# Daemon management
# ---------------------------------------------------------------------------

def _daemon_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False


def _ensure_daemon():
    global _daemon_started
    if _daemon_started and _daemon_running():
        return
    if not _daemon_running():
        log_path = CONFIG_DIR / "daemon.log"
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")
        subprocess.Popen(
            [sys.executable, "-m", "heylux.daemon"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        # Wait for socket
        for _ in range(30):
            if SOCKET_PATH.exists():
                break
            time.sleep(0.5)
    _daemon_started = True


# ---------------------------------------------------------------------------
# Voice pipeline
# ---------------------------------------------------------------------------

def _load_voice():
    """Lazy-load voice dependencies."""
    global _voice_loaded
    if _voice_loaded:
        return
    from heylux.voice import ensure_model
    ensure_model()
    _voice_loaded = True


def _listen_for_wake() -> str | None:
    """Listen for 'Hey Lux' + command. Returns command text or None."""
    from heylux.voice import listen_for_wake_command
    return listen_for_wake_command()


def _listen_for_command() -> str | None:
    """Listen for a follow-up command (no wake word needed)."""
    from heylux.voice import listen_once
    return listen_once()


def _send_to_daemon(prompt: str) -> str:
    """Send a prompt to the daemon and collect the text response."""
    import asyncio

    async def _send():
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        # Send with voice flag
        writer.write(
            json.dumps({"prompt": prompt, "voice": True}).encode() + b"\n"
        )
        await writer.drain()

        text_blocks = []
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=120)
            if not line:
                break
            msg = json.loads(line.decode())
            if msg["type"] == "text":
                text_blocks.append(msg["text"])
            elif msg["type"] == "done":
                break

        writer.close()
        await writer.wait_closed()
        return " ".join(text_blocks)

    return asyncio.run(_send())


def _speak(text: str):
    """Speak text using Edge TTS."""
    from heylux.voice import speak
    speak(text)


def _wait_for_speech():
    """Wait for TTS to finish."""
    from heylux.voice import wait_for_speech
    wait_for_speech()


# ---------------------------------------------------------------------------
# macOS notification helper
# ---------------------------------------------------------------------------

def _notify(title: str, subtitle: str = "", sound: bool = False):
    """Show a macOS notification."""
    try:
        rumps.notification(
            title=title,
            subtitle=subtitle,
            message="",
            sound=sound,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The menubar app
# ---------------------------------------------------------------------------

class HeyLuxApp(rumps.App):
    def __init__(self):
        super().__init__(
            name="Hey Lux",
            title=ICON_IDLE,
            quit_button="Quit Hey Lux",
        )
        self.menu = [
            rumps.MenuItem("Listening for 'Hey Lux'...", callback=None),
            None,  # separator
            rumps.MenuItem("Start Daemon", callback=self.on_start_daemon),
            rumps.MenuItem("Stop Daemon", callback=self.on_stop_daemon),
        ]
        self._wake_thread = None
        self._running = True

    def on_start_daemon(self, _):
        _ensure_daemon()
        _notify("Hey Lux", "Daemon started")

    def on_stop_daemon(self, _):
        if _daemon_running():
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            _notify("Hey Lux", "Daemon stopped")

    @rumps.clicked("Listening for 'Hey Lux'...")
    def on_status_click(self, _):
        pass  # non-interactive status item

    def _set_status(self, icon: str):
        """Update the menubar icon."""
        self.title = icon

    def _voice_loop(self):
        """Background thread: wake word detection → command → execute → repeat."""
        # Pre-load voice model and daemon at startup so first command is fast
        try:
            self._set_status(ICON_PROCESSING)
            _notify("Hey Lux", "Loading voice model...")
            _load_voice()
            _ensure_daemon()
            self._set_status(ICON_IDLE)
            _notify("Hey Lux", "Ready! Say 'Hey Lux' to start.")
        except Exception as e:
            _notify("Hey Lux", f"Voice init failed: {e}")
            self._set_status(ICON_IDLE)
            return

        while self._running:
            try:
                self._set_status(ICON_IDLE)

                # Listen for wake word + command
                command = _listen_for_wake()

                if command is None:
                    continue

                if command == "":
                    # Just "Hey Lux" with no command — prompt
                    self._set_status(ICON_LISTENING)
                    _speak("Listening.")
                    _wait_for_speech()
                    text = _listen_for_command()
                    if not text:
                        continue
                    command = text

                # Execute the command
                self._set_status(ICON_PROCESSING)

                try:
                    response = _send_to_daemon(command)
                except Exception:
                    _speak("Sorry, I couldn't connect to the daemon.")
                    _wait_for_speech()
                    continue

                # Speak the response
                if response:
                    self._set_status(ICON_SPEAKING)
                    _speak(response)
                    _wait_for_speech()

                # Multi-turn: listen for follow-up
                while self._running:
                    self._set_status(ICON_LISTENING)
                    follow_up = _listen_for_command()
                    if follow_up:
                        self._set_status(ICON_PROCESSING)
                        try:
                            response = _send_to_daemon(follow_up)
                        except Exception:
                            break
                        if response:
                            self._set_status(ICON_SPEAKING)
                            _speak(response)
                            _wait_for_speech()
                    else:
                        # Silence — back to wake word mode
                        break

            except Exception as e:
                # Don't crash the loop
                time.sleep(1)

    def start_voice_loop(self):
        """Start the background voice thread after the app is running."""
        self._wake_thread = threading.Thread(
            target=self._voice_loop,
            daemon=True,
        )
        self._wake_thread.start()


def main():
    """Entry point for lux-app."""
    app = HeyLuxApp()

    # Start voice loop in background after a short delay
    # (rumps needs to be running first)
    def _delayed_start():
        time.sleep(1)
        app.start_voice_loop()

    threading.Thread(target=_delayed_start, daemon=True).start()

    app.run()


if __name__ == "__main__":
    main()
