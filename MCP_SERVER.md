# Fiat-Lux MCP Server

Fiat-Lux exposes its lighting tools as an MCP (Model Context Protocol) server. Any AI agent or MCP client can connect to control Philips Hue lights with circadian intelligence, weather awareness, and scheduling.

## Available Tools

### Hue Light Control

| Tool | Description |
|---|---|
| `pair_hue_bridge` | One-time pairing with a Hue Bridge. User must press the link button first. |
| `get_hue_status` | List all lights organized by room/group, with current state and available scenes. |
| `set_lights` | Control individual lights — brightness, color temperature, hue/saturation, on/off. |
| `set_group` | Control all lights in a room/group at once. |
| `activate_scene` | Activate a named Hue scene in a room. |
| `breathing_pulse` | Perform a breathing pulse on one or more lights — fade in/out in a chosen color. Saves and restores previous state. |

### Circadian Intelligence

| Tool | Description |
|---|---|
| `get_circadian_recommendation` | Get the optimal lighting for the current time based on chronobiology. Returns color temperature (Kelvin), brightness (%), active lights, and scientific rationale. Weather-aware: adjusts for cloud cover and actual sunrise/sunset. |

### Routines

| Tool | Description |
|---|---|
| `list_routines` | List all saved lighting routines. |
| `save_routine` | Create or update a named routine (e.g. "bedtime", "focus"). Defines which lights to turn on/off and their settings. |
| `delete_routine` | Remove a saved routine. |

### Scheduling

| Tool | Description |
|---|---|
| `schedule_transition` | Schedule a gradual lighting transition at a future time. Uses the Hue bridge's native transition for smooth ramps up to ~109 minutes. Survives daemon restarts. |
| `list_scheduled` | List all pending scheduled transitions. |
| `cancel_scheduled` | Cancel a scheduled transition by job ID. |

### Weather

| Tool | Description |
|---|---|
| `setup_weather` | Set up weather integration via Open-Meteo. Auto-detects location via IP geolocation (no permissions needed), or accepts manual coordinates. |
| `get_current_weather` | Get current conditions — cloud cover, weather description, sunrise/sunset, UV index, temperature. |
| `update_location` | Refresh location (e.g. when traveling). |

### Calendar Alerts

| Tool | Description |
|---|---|
| `setup_calendar_alerts` | Check/install icalBuddy, list available macOS calendars. |
| `save_calendar_config` | Select which calendars to monitor for meeting alerts. |
| `set_alert_lights` | Configure which lights pulse for meeting alerts. Defaults to all lights. |

### User Memory

| Tool | Description |
|---|---|
| `get_user_profile` | Retrieve the user's saved profile (preferences, room layout, etc). |
| `save_user_info` | Save a learned fact about the user for future sessions. |
| `forget_user_info` | Remove a piece of user info. |

## Connecting to the MCP Server

The Fiat-Lux daemon exposes its tools through the Claude Agent SDK's `create_sdk_mcp_server`. The server is named `fiat_lux` and all tool names are prefixed with `mcp__fiat_lux__` when accessed through the SDK.

### From another Claude Agent SDK client

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server

# Import the tools you want
from fiat_lux.mcp.hue import ALL_HUE_TOOLS
from fiat_lux.mcp.circadian import get_circadian_recommendation

# Create the MCP server
fiat_lux_server = create_sdk_mcp_server(
    name="fiat_lux",
    version="0.1.0",
    tools=[get_circadian_recommendation, *ALL_HUE_TOOLS],
)

# Use it in your agent options
options = ClaudeAgentOptions(
    system_prompt="You are an assistant with access to smart lighting.",
    mcp_servers={"fiat_lux": fiat_lux_server},
    allowed_tools=["mcp__fiat_lux__*"],
)
```

### Tool naming convention

All tools follow the pattern `mcp__fiat_lux__<tool_name>`:

```
mcp__fiat_lux__set_lights
mcp__fiat_lux__get_circadian_recommendation
mcp__fiat_lux__schedule_transition
mcp__fiat_lux__breathing_pulse
...
```

## Configuration

All state is stored in `~/.config/fiat_lux/`. The MCP tools read/write:

- `hue.json` — Bridge credentials
- `user.json` — User profile
- `routines.json` — Named presets
- `calendars.json` — Calendar alert config + alert light preferences
- `schedule.json` — Pending scheduled transitions
- `weather.json` — Location for weather data
- `weather_cache.json` — Cached weather (30 min TTL)
