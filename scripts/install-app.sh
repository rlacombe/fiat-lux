#!/bin/bash
# Install Hey Lux as a macOS menubar app
#
# Creates a symlink so you can run: lux-app
# Or add it to Login Items to start on boot.

set -e

echo "Installing Hey Lux menubar app..."

# Ensure GUI deps are installed
cd "$(dirname "$0")/.."
uv sync --extra gui

# Create symlink
LUX_APP=$(uv run which lux-app 2>/dev/null || echo "")
if [ -n "$LUX_APP" ]; then
    mkdir -p ~/.local/bin
    ln -sf "$LUX_APP" ~/.local/bin/lux-app
    echo "Installed: lux-app"
    echo ""
    echo "To start: lux-app"
    echo "To auto-start on login: add lux-app to System Settings → Login Items"
else
    echo "Error: lux-app not found after install"
    exit 1
fi
