# Hey Lux!

**Voice-controlled smart lighting powered by circadian science and Claude**

*"Hey Lux, turn my lights to coding mode."*

Meet **Lux** — an AI lighting assistant for Philips Hue that understands circadian science, adapts to weather, and responds to your voice. Say "Hey Lux" and talk naturally — Lux listens, acts, and talks back.

<p align="center">
  <img src="docs/terminal.svg" alt="Lux terminal conversation" width="680">
</p>

## Why?

Most smart lighting is dumb. You get an app with color wheels and timers, but no understanding of *why* light matters. The science is clear: light is the primary signal that sets your circadian clock. The wrong light at the wrong time disrupts sleep, tanks focus, and affects mood. The right light — the right spectrum, intensity, and timing — does the opposite.

Lux brings that science into your room. It knows about melanopsin sensitivity, cortisol awakening response, melatonin suppression curves, and dim-light melatonin onset. It tracks actual sunrise and sunset for your location, adjusts for cloud cover, and shifts its recommendations with the seasons. And because it's powered by Claude, you can just *talk* to it — no manual programming, no fiddling with color pickers.

## What can Lux do?

Here are some real conversations:

```
You: wake me up with a sunrise tomorrow at 7:30

Lux: Scheduled! Starting at 7:30 AM, your lights will ramp from 1% warm amber
     (2000K) to 100% daylight (5500K) over 20 minutes. The Hue bridge handles
     the smooth fade — it'll run even if your laptop is asleep.
```

```
You: it's overcast today, my room feels dark

Lux: I can see that — 73% cloud cover right now. I've already boosted your
     circadian brightness 22% to compensate. Want me to push it higher?
```

```
You: candle on night stand 10m

Lux: Candle mode started on night stand, fading out over 10min.
```

```
You: I'm reading in bed, can you set the light so it doesn't mess with my sleep?

Lux: Night stand set to deep amber (2200K) at 40%. At this color temperature,
     melanopic content is near zero — your melatonin production stays untouched.
```

```
[Voice mode]

You: Hey Lux, switch to coding mode.
Lux: Setting coding mode now.
     → set_lights (x4)
Lux: Done, purple on desk and library, green on the rest.
```

## Features

- **Voice control** — say "Hey Lux" to wake, give a command, hear the response. Local Whisper STT, neural TTS, wake word detection. No cloud mic required.
- **Instant commands** — "lights off", "bedtime", "focus", "brighter" execute directly (<1s)
- **Named routines** — "bedtime", "morning", "focus", "coding" — customizable through conversation
- **Ambient modes** — "candle" for flickering candlelight, "breathe" for slow breathing glow
- **Natural language** — "make it Rilakkuma-colored", "sunset in my room" via Claude
- **Circadian automation** — time-based lighting grounded in melanopsin sensitivity and melatonin research
- **Weather-adaptive** — boosts brightness on cloudy days, shifts circadian curve to actual sunrise/sunset
- **Scheduling** — "sunrise at 8am tomorrow" — gradual transitions that run while you sleep
- **Calendar alerts** — synchronized light pulses before meetings (slow amber wave at T-5min, fast blue chirp at T-15s)
- **Persistent daemon** — boots once, stays warm, every command after that is fast
- **User memory** — Lux learns your name, room layout, sleep habits across sessions
- **MCP server** — other AI agents can plug into Lux's lighting tools
- **Built-in Hue control** — no external MCP servers needed

## Architecture

Lux runs as a daemon with three interaction modes:

```
              ┌─ Text REPL (lux)
User input ───┤
              ├─ Voice ("Hey Lux")  →  Whisper STT  →  daemon  →  Edge TTS
              │
              └─ One-shot (lux "lights off")
                        │
                        ▼
                     Daemon
                        │
                ├─ Shortcuts (regex + routines)  →  direct phue  < 1s
                │
                └─ Claude (persistent session)   →  tool calls   ~ 5s
                │
                └─ Background: calendar alerts, scheduler, ambient modes
```

**Shortcuts** pattern-match common commands, named routines, and ambient modes, executing directly via phue. No LLM, no network latency.

**Claude** handles everything else via a persistent `ClaudeSDKClient` session. The daemon boots the Claude Code process once and keeps it warm — subsequent messages skip the cold start. Voice mode uses Haiku for faster responses.

**Background tasks** run in the daemon's async event loop: calendar alerts poll every 30s, the scheduler checks for due transitions every 10s, and ambient modes (candle/breathing) run continuous light animations.

## Quickstart

```bash
git clone https://github.com/rlacombe/heylux.git
cd heylux
uv sync

# Start Lux
uv run lux start

# Talk to Lux
uv run lux "hello"
uv run lux setup          # guided Hue Bridge pairing
```

Requires a Claude Code subscription (Pro/Max). No API key needed if you're logged in.

To avoid typing `uv run` every time, create a symlink:

```bash
ln -s $(uv run which lux) ~/.local/bin/lux
```

Make sure `~/.local/bin` is in your `PATH`.

### Voice mode setup

Voice requires additional dependencies (Whisper + sounddevice):

```bash
uv sync --extra voice
lux listen                 # say "Hey Lux" followed by a command
```

First run downloads the Whisper base model (~140MB). Requires an arm64 Python on Apple Silicon (`uv python install 3.12`).

## Usage

### Text mode

The main text interface is the **interactive REPL** — just type `lux` and talk:

```bash
lux                        # start a conversation with Lux
```

Type `listen` or `voice` inside the REPL to switch to mic input.

### Voice mode

```bash
lux listen                 # always-on: "Hey Lux" wake word
lux --voice                # same as listen
```

Say "Hey Lux" followed by your command. Lux transcribes locally with Whisper, executes, and responds aloud with a neural voice. Supports multi-turn — keep talking after a response, or pause to return to wake word mode.

### All commands

```bash
# CLI
lux --help                 # show all commands
lux --version              # show version

# Daemon management
lux start                  # start the daemon
lux stop                   # stop it
lux status                 # check if running
lux restart                # restart

# Instant shortcuts (< 1s)
lux "lights off"
lux "lights on"
lux "brighter"
lux "dimmer"
lux "50%"
lux "circadian"

# Routines (< 1s, customizable)
lux "bedtime"              # nightstand only, warm
lux "morning"              # ceiling + desk, cool bright
lux "focus"                # ceiling + desk, peak alertness
lux "coding"               # your custom coding setup
lux "reading"              # nightstand + desk, warm white
lux "relax"                # lantern + nightstand, low amber
lux "goodnight"            # everything off
lux "routines"             # list all routines

# Ambient modes (continuous, say "stop" to end)
lux "candle"               # flickering candlelight on all lights
lux "candle on night stand" # candle on a specific light
lux "candle on night stand 10m"  # candle that fades out over 10 min
lux "breathe"              # slow breathing glow
lux "breathe on night stand" # breathing on a specific light

# Scheduling
lux "schedule a sunrise at 8am"  # gradual wake-up ramp

# Weather
lux "setup weather"        # auto-detect location, connect Open-Meteo

# Calendar alerts
lux setup calendar         # choose which calendars to monitor

# Natural language (~ 5s, via Claude)
lux "make it cozy"
lux "sunset in my room"
lux "update my bedtime routine to keep the lantern on"
```

## Configuration

Lux stores everything in `~/.config/heylux/`:

| File | Purpose |
|---|---|
| `hue.json` | Bridge IP and API credentials |
| `user.json` | User profile and preferences |
| `routines.json` | Named lighting presets |
| `calendars.json` | Calendars to monitor + alert light preferences |
| `schedule.json` | Pending scheduled transitions |
| `weather.json` | Location for weather data |
| `weather_cache.json` | Cached weather (refreshed every 30 min) |
| `light_map.json` | Circadian zone → light name mapping |
| `voice.json` | Voice model configuration |
| `history` | CLI command history |
| `lux.sock` | Daemon Unix socket |
| `daemon.log` | Daemon log output |

## MCP Server

Hey Lux exposes all its lighting tools as an MCP server in the `heylux/mcp/` package. Any AI agent can plug in and control your lights with circadian intelligence, weather awareness, and scheduling — no daemon required.

```python
from heylux.mcp.hue import ALL_HUE_TOOLS
from heylux.mcp.circadian import get_circadian_recommendation
from heylux.mcp.weather_tools import ALL_WEATHER_TOOLS
```

See **[MCP_SERVER.md](MCP_SERVER.md)** for the full tool reference (20+ tools) and integration guide.

## Requirements

- Python 3.12+ (arm64 on Apple Silicon for voice mode)
- [uv](https://docs.astral.sh/uv/)
- A Philips Hue Bridge + Hue bulbs
- Claude Code subscription (Pro or Max plan)
- For voice: `uv sync --extra voice` (adds Whisper + sounddevice)

## Disclaimer

> [!WARNING]
> Lux's circadian lighting recommendations are based on published photobiology and chronobiology research, but **Lux is not a medical device and does not provide medical advice**. Light exposure can affect circadian rhythms, sleep, and mood — if you have concerns about sleep disorders, seasonal affective disorder, or other health conditions, please consult a qualified healthcare professional. Lux is a tool for wellness and convenience, not a substitute for medical guidance.

## License

MIT
