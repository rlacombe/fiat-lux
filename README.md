# Fresnel

A chronobiology-powered lighting framework for Philips Hue. Meet **Lux**, your lighting scientist.

Fresnel is the framework. **Lux** is the agent — a circadian lighting specialist who manages your Philips Hue lights for better focus, energy, and sleep, grounded in real photobiology research.

## Why Fresnel?

[Augustin-Jean Fresnel](https://en.wikipedia.org/wiki/Augustin-Jean_Fresnel) (1788–1827) transformed our understanding of light and saved countless lives in the process. Against fierce opposition from the particle theory establishment, he championed the wave theory of light — developing the mathematics of diffraction and interference that proved light behaves as a wave. His work on birefringence and polarization gave us tools to understand how light interacts with matter at a fundamental level.

But Fresnel's most tangible legacy is the **Fresnel lens**: a brilliant optical design that made lighthouse beams visible from over 30 kilometers away. Before his invention, shipwrecks from poor coastal visibility were routine. The Fresnel lens turned dim lighthouse flames into life-saving beacons and remains in use nearly two centuries later.

This framework carries his name because it shares his mission: **using the physics of light to make a practical difference in people's lives** — whether that's guiding ships safely home or tuning your bedroom lights to protect your sleep.

## Features

- **Instant commands** — "lights off", "bedtime", "focus", "brighter" execute directly (<1s)
- **Named routines** — "bedtime", "morning", "focus", "relax" — customizable through conversation
- **Natural language** — "make it Rilakkuma-colored", "sunset in my room" via Claude
- **Circadian automation** — time-based lighting grounded in melanopsin sensitivity and melatonin research
- **Persistent daemon** — boots once, stays warm, every command after that is fast
- **User memory** — Lux learns your name, room layout, sleep habits across sessions
- **Built-in Hue control** — no external MCP servers needed

## Architecture

Lux runs as a daemon with two execution paths:

```
User input → CLI
               │
               ├─ Shortcuts (regex + routines)  →  direct phue  < 1s
               │
               └─ Claude (persistent session)   →  tool calls   ~ 5s
```

**Shortcuts** pattern-match common commands and named routines, executing directly via phue. No LLM, no network latency.

**Claude** handles everything else via a persistent `ClaudeSDKClient` session. The daemon boots the Claude Code process once and keeps it warm — subsequent messages skip the cold start.

## Quickstart

```bash
git clone https://github.com/rlacombe/fresnel.git
cd fresnel
uv sync

# Start Lux
uv run fresnel start

# Talk to Lux
uv run fresnel "hello"
uv run fresnel setup          # guided Hue Bridge pairing
```

Requires a Claude Code subscription (Pro/Max). No API key needed if you're logged in.

## Usage

```bash
# Daemon management
uv run fresnel start           # start the daemon
uv run fresnel stop            # stop it
uv run fresnel status          # check if running
uv run fresnel restart         # restart

# Interactive mode
uv run fresnel                 # REPL

# Instant shortcuts (< 1s)
uv run fresnel "lights off"
uv run fresnel "lights on"
uv run fresnel "brighter"
uv run fresnel "dimmer"
uv run fresnel "50%"
uv run fresnel "circadian"

# Routines (< 1s, customizable)
uv run fresnel "bedtime"       # nightstand + lantern, warm
uv run fresnel "morning"       # ceiling + desk, cool bright
uv run fresnel "focus"         # ceiling + desk, peak alertness
uv run fresnel "reading"       # nightstand + desk, warm white
uv run fresnel "relax"         # lantern + nightstand, low amber
uv run fresnel "goodnight"     # everything off
uv run fresnel "routines"      # list all routines

# Natural language (~ 5s, via Claude)
uv run fresnel "make it cozy"
uv run fresnel "sunset in my room"
uv run fresnel "update my bedtime routine to keep the lantern on"
```

## Configuration

Lux stores everything in `~/.config/fresnel/`:

| File | Purpose |
|---|---|
| `hue.json` | Bridge IP and API credentials |
| `user.json` | User profile and preferences |
| `routines.json` | Named lighting presets |
| `fresnel.sock` | Daemon Unix socket |
| `daemon.log` | Daemon log output |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Philips Hue Bridge + Hue bulbs
- Claude Code subscription (Pro or Max plan)

## License

MIT
