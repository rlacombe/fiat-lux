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

console = Console()

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
SOCKET_PATH = CONFIG_DIR / "lux.sock"
PID_FILE = CONFIG_DIR / "lux.pid"
HISTORY_FILE = CONFIG_DIR / "history"

SEND_TIMEOUT = 30  # seconds


def _version() -> str:
    """Get package version."""
    from importlib.metadata import version

    try:
        return version("fiat-lux")
    except Exception:
        return "dev"


HELP_TEXT = """\
[bold]Lux[/bold] — chronobiology-powered lighting assistant

[bold]Usage:[/bold]
  lux                          Interactive mode
  lux "<command>"              One-shot command
  lux start                    Start the daemon
  lux stop                     Stop the daemon
  lux status                   Check daemon status
  lux restart                  Restart the daemon
  lux setup calendar           Configure calendar alerts
  lux --help, -h               Show this help
  lux --version, -v            Show version

[bold]Examples:[/bold]
  lux "lights off"             Turn all lights off
  lux "circadian"              Apply circadian lighting
  lux "50%"                    Set brightness to 50%
  lux "focus"                  Activate focus routine
  lux "breathe"                Start breathing light mode
  lux "candle"                 Start candle flicker mode
  lux "make it cozy"           Ask Lux (uses AI)

[bold]Shortcuts:[/bold]
  on/off, brighter/dimmer, circadian, breathe/candle/stop,
  and any saved routine name are handled instantly (<1s).
  Everything else goes through Lux's AI for natural language control.
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
        console.print("[dim]Daemon already running.[/dim]")
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
    with console.status("[bold cyan]Starting Lux...", spinner="dots"):
        for _ in range(30):
            if SOCKET_PATH.exists():
                break
            if proc.poll() is not None:
                console.print(
                    f"[red]Daemon exited with code {proc.returncode}. "
                    f"Check {log_path}[/red]"
                )
                return
            time.sleep(0.5)
        else:
            console.print(f"[red]Daemon failed to start. Check {log_path}[/red]")
            return

    console.print("[dim]Lux is ready.[/dim]")


def _stop_daemon() -> None:
    """Stop the daemon."""
    if not _daemon_running():
        console.print("[dim]Daemon not running.[/dim]")
        return

    pid = int(PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    console.print("[dim]Daemon stopped.[/dim]")


async def _send_to_daemon(prompt: str) -> None:
    """Send a prompt to the daemon and stream the response."""
    reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))

    try:
        writer.write(json.dumps({"prompt": prompt}).encode() + b"\n")
        await writer.drain()

        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=SEND_TIMEOUT)
            if not line:
                break

            msg = json.loads(line.decode())

            if msg["type"] == "text":
                console.print(Markdown(msg["text"]))
            elif msg["type"] == "tool":
                console.print(f"[dim]-> {msg['name']}[/dim]")
            elif msg["type"] == "error":
                console.print(f"[red]Error: {msg['text']}[/red]")
            elif msg["type"] == "done":
                break
    except asyncio.TimeoutError:
        console.print("[red]Daemon not responding (timed out after 30s).[/red]")
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
        console.print("[yellow]Connection lost. Restarting daemon...[/yellow]")
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
        "[bold]Lux[/bold] -- your chronobiology-powered lighting assistant\n"
        "[dim]Type a command, or 'quit' to exit.[/dim]\n"
    )

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

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
        console.print(f"fiat-lux {_version()}")
    elif args[0] == "start":
        _start_daemon()
    elif args[0] == "stop":
        _stop_daemon()
    elif args[0] == "status":
        if _daemon_running():
            pid = PID_FILE.read_text().strip()
            console.print(f"[green]Daemon running (pid {pid})[/green]")
        else:
            console.print("[yellow]Daemon not running[/yellow]")
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
    else:
        prompt = " ".join(args)
        _send(prompt)


if __name__ == "__main__":
    main()
