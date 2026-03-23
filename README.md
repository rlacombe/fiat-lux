# Fresnel

A chronobiology-powered lighting agent for Philips Hue, built for speed.

Fresnel manages your Philips Hue lights based on circadian science — it knows when to energize you with cool bright light, when to wind you down with warm amber, and how to protect your sleep. Named after [Augustin-Jean Fresnel](https://en.wikipedia.org/wiki/Augustin-Jean_Fresnel), who revolutionized our understanding of light.

## Features

- **Instant common commands** — "lights off", "circadian", "50%" execute directly, no LLM needed (<0.5s)
- **Natural language control** — "make it cozy", "Rilakkuma-colored", "wind me down for bed" via any LLM
- **Circadian automation** — time-based lighting grounded in melanopsin sensitivity and melatonin research
- **Built-in Hue control** — no external MCP servers. Bridge pairing, lights, groups, scenes all built in
- **User memory** — learns your name, room layout, sleep habits across sessions
- **Any LLM backend** — LM Studio (local/free), OpenRouter (cheap), OpenAI, or Claude Agent SDK

## Architecture

Fresnel uses a three-tier execution model, fastest first:

```
User input
    │
    ├─ Tier 1: Shortcuts (regex)     → direct phue call     < 0.5s   free
    ├─ Tier 2: LLM engine (OpenAI)   → tool-use loop        ~ 2-5s   cheap/free
    └─ Tier 3: Agent SDK (Claude)    → full Claude Code      ~ 14s    subscription
```

**Tier 1** pattern-matches common commands ("lights off", "circadian", "50%") and executes directly via phue. No network, no LLM.

**Tier 2** sends the prompt to any OpenAI-compatible API (LM Studio, OpenRouter, etc.) with tool definitions. The LLM picks the right tool, Fresnel executes it. One HTTP call.

**Tier 3** falls back to the Claude Agent SDK for complex multi-step tasks like initial bridge setup. This spawns a full Claude Code process — powerful but slow.

## Quickstart

```bash
git clone https://github.com/rlacombe/fresnel.git
cd fresnel
uv sync
cp .env.example .env
```

### Option A: LM Studio (local, free)

1. Install [LM Studio](https://lmstudio.ai) and download a model (e.g., Qwen 2.5 7B Instruct)
2. Start the local server (LM Studio → Local Server → Start)
3. Edit `.env`:
   ```
   FRESNEL_BASE_URL=http://localhost:1234/v1
   FRESNEL_API_KEY=lm-studio
   FRESNEL_MODEL=qwen2.5-7b-instruct
   ```
4. `uv run fresnel "hello"`

### Option B: OpenRouter (cheap, any model)

1. Get an API key at [openrouter.ai](https://openrouter.ai)
2. Edit `.env`:
   ```
   FRESNEL_BASE_URL=https://openrouter.ai/api/v1
   FRESNEL_API_KEY=sk-or-...
   FRESNEL_MODEL=anthropic/claude-3-haiku
   ```
3. `uv run fresnel "hello"`

### Option C: Claude Agent SDK (subscription)

1. Set `FRESNEL_ENGINE=agent-sdk` and `ANTHROPIC_API_KEY` in `.env`
2. `uv run fresnel "hello"`

## Usage

```bash
# Interactive mode
uv run fresnel

# One-shot commands
uv run fresnel "turn my lights off"          # → shortcut, instant
uv run fresnel "50%"                         # → shortcut, instant
uv run fresnel circadian                     # → shortcut, instant
uv run fresnel "make it Rilakkuma-colored"   # → LLM engine
uv run fresnel setup                         # → Agent SDK (guided)
```

## Configuration

Fresnel stores config in `~/.config/fresnel/`:
- `hue.json` — Bridge IP and API credentials
- `user.json` — User profile and preferences

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Philips Hue Bridge + Hue bulbs
- One of: LM Studio, OpenRouter API key, Anthropic API key, or OpenAI API key

## License

MIT
