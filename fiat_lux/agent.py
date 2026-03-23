"""Fiat-Lux CLI — thin client that talks to the Lux daemon.

Usage:
    lux                          Interactive mode
    lux "make it cozy"           One-shot command
    lux start                    Start the daemon
    lux stop                     Stop the daemon
    lux status                   Check daemon status
    lux restart                  Restart the daemon
    lux setup calendar           Configure calendar alerts
    lux --help                   Show this help
    lux --version                Show version
"""

import asyncio
import json
import os
import readline
import signal
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
from rich.theme import Theme

# Tokyo Night color scheme — matches the README terminal illustration
THEME = Theme({
    "lux.user": "bold #7dcfff",        # cyan — user prompt
    "lux.label": "bold #9ece6a",        # green — Lux label
    "lux.text": "#a9b1d6",              # muted blue-white — response text
    "lux.highlight": "#e0af68",         # amber — values and highlights
    "lux.science": "italic #7aa2f7",    # blue italic — science notes
    "lux.dim": "#565f89",               # gray — secondary info
    "lux.tool": "#565f89",              # gray — tool call indicators
    "lux.error": "#f7768e",             # red — errors
    "lux.success": "#9ece6a",           # green — success messages
    "lux.warn": "#e0af68",              # amber — warnings
    "lux.title": "bold #c0caf5",        # bright white — titles
})

console = Console(theme=THEME)

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
SOCKET_PATH = CONFIG_DIR / "lux.sock"
PID_FILE = CONFIG_DIR / "lux.pid"
HISTORY_FILE = CONFIG_DIR / "history"

SEND_TIMEOUT = 120  # seconds — breathing pulses and tool calls can take a while


def _version() -> str:
    """Get package version."""
    from importlib.metadata import version

    try:
        return version("fiat-lux")
    except Exception:
        return "dev"


HELP_TEXT = """\
[lux.title]Lux[/lux.title] [lux.dim]—[/lux.dim] [lux.text]chronobiology-powered lighting assistant[/lux.text]

[lux.title]Usage:[/lux.title]
  [lux.highlight]lux[/lux.highlight]                          Interactive mode
  [lux.highlight]lux "<command>"[/lux.highlight]              One-shot command
  [lux.highlight]lux start[/lux.highlight]                    Start the daemon
  [lux.highlight]lux stop[/lux.highlight]                     Stop the daemon
  [lux.highlight]lux status[/lux.highlight]                   Check daemon status
  [lux.highlight]lux restart[/lux.highlight]                  Restart the daemon
  [lux.highlight]lux setup calendar[/lux.highlight]           Configure calendar alerts
  [lux.highlight]lux setup weather[/lux.highlight]            Connect weather data
  [lux.highlight]lux listen[/lux.highlight]                   Voice command (one-shot)
  [lux.highlight]lux --voice, -V[/lux.highlight]              Voice REPL (continuous)
  [lux.highlight]lux --help, -h[/lux.highlight]               Show this help
  [lux.highlight]lux --version, -v[/lux.highlight]            Show version

[lux.title]Examples:[/lux.title]
  [lux.highlight]lux "lights off"[/lux.highlight]             Turn all lights off
  [lux.highlight]lux "circadian"[/lux.highlight]              Apply circadian lighting
  [lux.highlight]lux "50%"[/lux.highlight]                    Set brightness to 50%
  [lux.highlight]lux "focus"[/lux.highlight]                  Activate focus routine
  [lux.highlight]lux "breathe"[/lux.highlight]                Start breathing light mode
  [lux.highlight]lux "candle"[/lux.highlight]                 Start candle flicker mode
  [lux.highlight]lux "make it cozy"[/lux.highlight]           Ask Lux (uses AI)

[lux.title]Shortcuts (in REPL):[/lux.title]
  [lux.dim]on/off, brighter/dimmer, circadian, breathe/candle/stop,
  and any saved routine name are handled instantly (<1s).
  Everything else goes through Lux's AI for natural language control.

  Note: "lux stop" controls the daemon. Type "stop" inside the
  REPL to stop ambient modes (candle, breathing).[/lux.dim]
"""


def _daemon_running() -> bool:
    """Check if the daemon is alive."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = check if process exists
        return True
    except (ProcessLookupError, ValueError):
        # Stale PID file
        PID_FILE.unlink(missing_ok=True)
        return False


def _start_daemon() -> None:
    """Start the daemon in the background with a spinner."""
    if _daemon_running():
        console.print("[lux.dim]Daemon already running.[/lux.dim]")
        return

    # Start daemon as a background process
    log_path = CONFIG_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")

    proc = subprocess.Popen(
        [sys.executable, "-m", "fiat_lux.daemon"],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    # Wait for socket with a spinner
    with console.status("[lux.highlight]Starting Lux...", spinner="dots"):
        for _ in range(30):
            if SOCKET_PATH.exists():
                break
            if proc.poll() is not None:
                console.print(
                    f"[lux.error]Daemon exited with code {proc.returncode}. "
                    f"Check {log_path}[/lux.error]"
                )
                return
            time.sleep(0.5)
        else:
            console.print(f"[lux.error]Daemon failed to start. Check {log_path}[/lux.error]")
            return

    console.print("[lux.success]Lux is ready.[/lux.success]")


def _stop_daemon() -> None:
    """Stop the daemon and clean up stale files."""
    if not _daemon_running():
        # Clean up stale socket if process is gone
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink(missing_ok=True)
        console.print("[lux.dim]Daemon not running.[/lux.dim]")
        return

    pid = int(PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)

    # Wait for process to actually die and clean up
    for _ in range(20):
        try:
            os.kill(pid, 0)
            time.sleep(0.25)
        except ProcessLookupError:
            break

    # Clean up stale files the daemon may not have removed
    PID_FILE.unlink(missing_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    console.print("[lux.dim]Daemon stopped.[/lux.dim]")


async def _send_to_daemon(prompt: str) -> None:
    """Send a prompt to the daemon and stream the response."""
    reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))

    try:
        writer.write(json.dumps({"prompt": prompt}).encode() + b"\n")
        await writer.drain()

        first_text = True
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=SEND_TIMEOUT)
            if not line:
                break

            msg = json.loads(line.decode())

            if msg["type"] == "text":
                if first_text:
                    console.print()
                    console.print("[lux.label]Lux:[/lux.label]")
                    first_text = False
                console.print(
                    Markdown(msg["text"]),
                    style="lux.text",
                )
            elif msg["type"] == "tool":
                console.print(f"[lux.tool]  -> {msg['name']}[/lux.tool]")
            elif msg["type"] == "error":
                console.print(f"[lux.error]Error: {msg['text']}[/lux.error]")
            elif msg["type"] == "done":
                break
    except asyncio.TimeoutError:
        console.print("[lux.error]Daemon not responding (timed out after 30s).[/lux.error]")
    finally:
        writer.close()
        await writer.wait_closed()


def _send_with_tts(prompt: str, speak_fn) -> None:
    """Send a prompt and speak the response aloud."""
    if not _daemon_running():
        _start_daemon()

    try:
        asyncio.run(_send_to_daemon_tts(prompt, speak_fn))
    except (ConnectionRefusedError, FileNotFoundError):
        console.print("[lux.warn]Connection lost. Restarting daemon...[/lux.warn]")
        _stop_daemon()
        time.sleep(0.5)
        _start_daemon()
        asyncio.run(_send_to_daemon_tts(prompt, speak_fn))


async def _send_to_daemon_tts(prompt: str, speak_fn) -> None:
    """Send prompt, display response, and speak each text block as it arrives."""
    reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))

    try:
        writer.write(json.dumps({"prompt": prompt}).encode() + b"\n")
        await writer.drain()

        first_text = True
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=SEND_TIMEOUT)
            if not line:
                break

            msg = json.loads(line.decode())

            if msg["type"] == "text":
                if first_text:
                    console.print()
                    console.print("[lux.label]Lux:[/lux.label]")
                    first_text = False
                console.print(Markdown(msg["text"]), style="lux.text")
                # Speak immediately — runs in background thread
                speak_fn(msg["text"])
            elif msg["type"] == "tool":
                console.print(f"[lux.tool]  -> {msg['name']}[/lux.tool]")
            elif msg["type"] == "error":
                console.print(f"[lux.error]Error: {msg['text']}[/lux.error]")
            elif msg["type"] == "done":
                break

    except asyncio.TimeoutError:
        console.print("[lux.error]Daemon not responding (timed out).[/lux.error]")
    finally:
        writer.close()
        await writer.wait_closed()


def _send(prompt: str) -> None:
    """Send a prompt, auto-starting the daemon if needed."""
    if not _daemon_running():
        _start_daemon()

    try:
        asyncio.run(_send_to_daemon(prompt))
    except (ConnectionRefusedError, FileNotFoundError):
        console.print("[lux.warn]Connection lost. Restarting daemon...[/lux.warn]")
        _stop_daemon()
        time.sleep(0.5)
        _start_daemon()
        asyncio.run(_send_to_daemon(prompt))


def _setup_readline() -> None:
    """Configure readline with persistent history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(HISTORY_FILE))
    except FileNotFoundError:
        pass
    readline.set_history_length(500)


def _save_readline() -> None:
    """Persist readline history to disk."""
    try:
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass


def _interactive() -> None:
    """Interactive REPL — all messages go through the daemon."""
    if not _daemon_running():
        _start_daemon()

    _setup_readline()

    console.print(
        "[lux.title]Lux[/lux.title] [lux.dim]--[/lux.dim] "
        "[lux.text]your chronobiology-powered lighting assistant[/lux.text]\n"
        "[lux.dim]Type a command, 'listen' for voice, or 'quit' to exit.[/lux.dim]\n"
    )

    try:
        while True:
            try:
                user_input = console.input("[lux.user]You:[/lux.user] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[lux.dim]Goodbye![/lux.dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[lux.dim]Goodbye![/lux.dim]")
                break

            # Voice input from within the REPL
            if user_input.lower() in ("listen", "voice"):
                _do_voice_in_repl()
                continue

            readline.add_history(user_input)

            console.print()
            _send(user_input)
            console.print()
    finally:
        _save_readline()


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args:
        _interactive()
        return

    if args[0] in ("--help", "-h"):
        console.print(HELP_TEXT)
    elif args[0] in ("--version", "-v"):
        console.print(f"[lux.title]fiat-lux[/lux.title] [lux.highlight]{_version()}[/lux.highlight]")
    elif args[0] == "start":
        _start_daemon()
    elif args[0] == "stop":
        _stop_daemon()
    elif args[0] == "status":
        if _daemon_running():
            pid = PID_FILE.read_text().strip()
            console.print(f"[lux.success]Daemon running (pid {pid})[/lux.success]")
        else:
            console.print("[lux.warn]Daemon not running[/lux.warn]")
    elif args[0] == "restart":
        _stop_daemon()
        time.sleep(1)
        _start_daemon()
    elif args[0] == "setup" and len(args) > 1 and args[1] == "calendar":
        from fiat_lux.calendar import setup_interactive

        setup_interactive()
    elif args[0] == "setup" and len(args) > 1 and args[1] == "weather":
        prompt = "Please set up weather integration. Ask me for permission before using location services."
        _send(prompt)
    elif args[0] == "listen":
        _listen_once()
    elif args[0] in ("--voice", "-V"):
        _voice_interactive()
    else:
        prompt = " ".join(args)
        _send(prompt)


def _load_voice_model():
    """Load Whisper model with a spinner. Returns (listen_once, speak) or (None, None)."""
    try:
        from fiat_lux.voice import ensure_model, listen_once, speak
        import fiat_lux.voice as voice_mod
        voice_mod._console = console
    except ImportError:
        console.print(
            "[lux.error]Voice dependencies not installed.[/lux.error]\n"
            "[lux.dim]Install with: uv sync --extra voice[/lux.dim]"
        )
        return None, None

    with console.status("[lux.highlight]Loading voice model...", spinner="dots"):
        try:
            ensure_model()
        except ImportError as e:
            console.print(f"[lux.error]{e}[/lux.error]")
            return None, None

    return listen_once, speak


def _do_voice_in_repl() -> None:
    """Handle a single voice command from within the REPL."""
    try:
        from fiat_lux.voice import ensure_model, listen_once, speak
        import fiat_lux.voice as voice_mod
        voice_mod._console = console
    except ImportError:
        console.print(
            "[lux.error]Voice deps not installed.[/lux.error] "
            "[lux.dim]Run: uv sync --extra voice[/lux.dim]\n"
        )
        return

    with console.status("[lux.highlight]Loading voice model...", spinner="dots"):
        try:
            ensure_model()
        except ImportError as e:
            console.print(f"[lux.error]{e}[/lux.error]\n")
            return

    console.print("[lux.highlight]Listening...[/lux.highlight]")
    try:
        text = listen_once()
    except ImportError as e:
        console.print(f"[lux.error]{e}[/lux.error]\n")
        return

    if text:
        console.print(f"\n[lux.user]You:[/lux.user] {text}\n")
        _send_with_tts(text, speak)
        from fiat_lux.voice import wait_for_speech
        wait_for_speech()
        console.print()
    else:
        console.print("[lux.dim]No speech detected.[/lux.dim]\n")


def _listen_once() -> None:
    """One-shot voice command: record → transcribe → send → speak."""
    listen_once, speak = _load_voice_model()
    if listen_once is None:
        return

    if not _daemon_running():
        _start_daemon()

    console.print("[lux.highlight]Listening...[/lux.highlight] (speak, then pause to send)")
    try:
        text = listen_once()
    except ImportError as e:
        console.print(f"[lux.error]{e}[/lux.error]")
        return
    if text:
        console.print(f"\n[lux.user]You:[/lux.user] {text}\n")
        _send_with_tts(text, speak)
        # Wait for Emma to finish speaking before exiting
        from fiat_lux.voice import wait_for_speech
        wait_for_speech()
    else:
        console.print("[lux.dim]No speech detected.[/lux.dim]")


def _voice_interactive() -> None:
    """Voice REPL — continuous listen loop with TTS responses."""
    result = _load_voice_model()
    if result[0] is None:
        return
    listen_once, speak = result

    if not _daemon_running():
        _start_daemon()

    console.print(
        "[lux.title]Lux[/lux.title] [lux.dim]--[/lux.dim] "
        "[lux.text]voice mode[/lux.text]\n"
        "[lux.dim]Speak a command. Press Ctrl+C to exit.[/lux.dim]\n"
    )

    while True:
        try:
            console.print("[lux.highlight]Listening...[/lux.highlight]")
            try:
                text = listen_once()
            except ImportError as e:
                console.print(f"[lux.error]{e}[/lux.error]")
                break
            if text:
                console.print(f"\n[lux.user]You:[/lux.user] {text}\n")
                _send_with_tts(text, speak)
                console.print()
            else:
                console.print("[lux.dim]...[/lux.dim]")
        except KeyboardInterrupt:
            console.print("\n[lux.dim]Goodbye![/lux.dim]")
            break


if __name__ == "__main__":
    main()
