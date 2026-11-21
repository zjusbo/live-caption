#!/usr/bin/env bash
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
# ./run.sh <video_file> [options]
#
# Options:
#   -o, --output <file>      Output .srt path (default: <video>.srt)
#   -m, --model  <size>      Whisper model: tiny|base|small|medium|large (default: large)
#   --batch-size <n>         Segments per Claude API call (default: 50)
#
# Examples:
#   ./run.sh video.mp4
#   ./run.sh video.mp4 -o subtitles.srt -m medium
#   ./run.sh video.mp4 --batch-size 30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -f "$PYTHON" ]]; then
  echo "Error: virtual environment not found at $VENV"
  echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <video_file> [-o output.srt] [-m tiny|base|small|medium|large] [--batch-size N]"
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/transcribe_to_srt.py" "$@"
