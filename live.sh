#!/usr/bin/env bash
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
# ./live.sh [options]          Live Chinese subtitles for whatever the Mac is playing
#
# Options:
#   --show-original            Also show the Japanese line
#   --font-size <pt>           Subtitle size (default: 34)
#   --lang <code|auto>         Spoken language (default: ja)
#   --file <media>             Caption a file instead of system audio (testing)
#   (./live.sh --help lists the rest)
#
# First run: allow your terminal app in System Settings → Privacy & Security →
# Screen & System Audio Recording, then restart the terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
SRC="$SCRIPT_DIR/live_captions_helper.swift"
HELPER="$SCRIPT_DIR/.build/live_captions_helper"

if [[ ! -f "$PYTHON" ]]; then
  echo "Error: virtual environment not found at $SCRIPT_DIR/.venv"
  echo "Run: python3 -m venv .venv && .venv/bin/pip install mlx-whisper anthropic python-dotenv"
  exit 1
fi

# (Re)build the native capture/overlay helper when the source changes
if [[ ! -x "$HELPER" || "$SRC" -nt "$HELPER" ]]; then
  echo "Building live_captions_helper …"
  mkdir -p "$SCRIPT_DIR/.build"
  swiftc -O -swift-version 5 "$SRC" -o "$HELPER"
fi

exec "$PYTHON" "$SCRIPT_DIR/live_captions.py" "$@"
