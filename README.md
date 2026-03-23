# Fiat-Lux

*"Let there be light."*

Meet **Lux** — a chronobiology-powered lighting agent for Philips Hue. Lux manages your lights for better focus, energy, and sleep, grounded in real photobiology research.

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
You: create a "movie" routine — just the lantern at 10%, deep red

Lux: Saved routine 'movie'. Trigger it anytime by typing 'movie'.
```

```
You: what should my lights be right now?

Lux: It's 3:15 PM — Afternoon mode. 6000K at 95% on ceiling and desk.
     This maintains the cortisol plateau and supports cognitive performance
     through the natural post-lunch dip. Sunset today is at 7:06 PM,
     so wind-down starts around 5:00.
```

## Features

- **Instant commands** — "lights off", "bedtime", "focus", "brighter" execute directly (<1s)
- **Named routines** — "bedtime", "morning", "focus", "relax" — customizable through conversation
- **Ambient modes** — "candle" for flickering candlelight, "breathe" for slow breathing glow
- **Natural language** — "make it Rilakkuma-colored", "sunset in my room" via Claude
- **Circadian automation** — time-based lighting grounded in melanopsin sensitivity and melatonin research
- **Weather-adaptive** — boosts brightness on cloudy days, shifts circadian curve to actual sunrise/sunset
- **Scheduling** — "sunrise at 8am tomorrow" — gradual transitions that run while you sleep
- **Calendar alerts** — breathing pulse on your desk lamp before meetings (amber at T-5min, blue at T-15s)
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

**Shortcuts** pattern-match common commands, named routines, and ambient modes, executing directly via phue. No LLM, no network latency.

**Claude** handles everything else via a persistent `ClaudeSDKClient` session. The daemon boots the Claude Code process once and keeps it warm — subsequent messages skip the cold start.

**Background tasks** run in the daemon's async event loop: calendar alerts poll every 30s, the scheduler checks for due transitions every 10s, and ambient modes (candle/breathing) run continuous light animations.

## Quickstart

```bash
git clone https://github.com/rlacombe/fiat-lux.git
cd fiat-lux
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

## Usage

```bash
# CLI
lux --help                 # show all commands
lux --version              # show version

# Daemon management
lux start                  # start the daemon
lux stop                   # stop it
lux status                 # check if running
lux restart                # restart

# Interactive mode (with readline history)
lux                        # REPL

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

Lux stores everything in `~/.config/fiat_lux/`:

| File | Purpose |
|---|---|
| `hue.json` | Bridge IP and API credentials |
| `user.json` | User profile and preferences |
| `routines.json` | Named lighting presets |
| `calendars.json` | Calendars to monitor for meeting alerts |
| `schedule.json` | Pending scheduled transitions |
| `weather.json` | Location for weather data |
| `weather_cache.json` | Cached weather (refreshed every 30 min) |
| `history` | CLI command history |
| `lux.sock` | Daemon Unix socket |
| `daemon.log` | Daemon log output |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Philips Hue Bridge + Hue bulbs
- Claude Code subscription (Pro or Max plan)

## Disclaimer

Lux's circadian lighting recommendations are based on published photobiology and chronobiology research, but **Lux is not a medical device and does not provide medical advice**. Light exposure can affect circadian rhythms, sleep, and mood — if you have concerns about sleep disorders, seasonal affective disorder, or other health conditions, please consult a qualified healthcare professional. Lux is a tool for wellness and convenience, not a substitute for medical guidance.

## License

MIT
