#!/usr/bin/env bash
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
# ./live.sh [options]          Live Chinese subtitles for whatever the Mac is playing
#
# Options:
#   --show-original            Also show the Japanese line
#   --translator claude        Translate with the Claude API (paid) instead of Apple
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

# ── Preflight: catch a missing prerequisite with a clear fix ─────────────────
fail() { echo "Error: $1" >&2; shift; for line in "$@"; do echo "  $line" >&2; done; exit 1; }

[[ "$(uname -m)" == "arm64" ]] || fail "live captions need a Mac with Apple Silicon (M1 or later)."
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
(( macos_major >= 15 )) || fail "live captions need macOS 15 or later (this Mac runs $(sw_vers -productVersion))."
command -v swiftc >/dev/null || fail "the Swift compiler is missing." "Install Apple's Command Line Tools: xcode-select --install"

if [[ ! -x "$PYTHON" ]]; then
  fail "Python environment not set up. From $SCRIPT_DIR run:" \
       "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || fail ".venv uses Python $("$PYTHON" -c 'import platform; print(platform.python_version())'); 3.10 or newer is needed." \
          "Install one (e.g. brew install python@3.13), then recreate it:" \
          "rm -rf .venv && /opt/homebrew/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
"$PYTHON" -c 'import mlx_whisper, anthropic, dotenv' 2>/dev/null \
  || fail "Python packages missing. Run: .venv/bin/pip install -r requirements.txt"

# (Re)build the native capture/overlay helper when the source changes
if [[ ! -x "$HELPER" || "$SRC" -nt "$HELPER" ]]; then
  echo "Building live_captions_helper …"
  mkdir -p "$SCRIPT_DIR/.build"
  swiftc -O -swift-version 5 "$SRC" -o "$HELPER" || fail "building the helper failed (see the compiler output above)."
fi

exec "$PYTHON" "$SCRIPT_DIR/live_captions.py" "$@"
