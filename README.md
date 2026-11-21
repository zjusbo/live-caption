# Live Caption

Real-time Simplified Chinese subtitles for whatever your Mac is playing: any app, full-screen video included. Speech recognition and translation both run on your Mac, so there's no API key, no per-use cost, and nothing is saved to disk. Optionally, `--translator claude` translates with the Claude API instead (paid).

## Requirements

- Mac with Apple Silicon (M1 or later), macOS 15 or later
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.10+, e.g. `brew install python@3.13` (the `python3` that ships with macOS is 3.9, too old)
- About 3 GB of disk: 1.3 GB for Python packages, 1.6 GB for the Whisper model

## Quick start

```bash
git clone https://github.com/zjusbo/live-caption.git
cd live-caption
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./live.sh
```

The first run compiles the native helper and downloads the Whisper model (about 1.6 GB, once). When **字幕已就绪** appears at the bottom of the screen, play something. Press Ctrl+C to stop.

**One-time macOS setup**

1. **Audio permission:** allow your terminal app (Terminal, iTerm, …) under **System Settings → Privacy & Security → Screen & System Audio Recording**, then quit and reopen the terminal. If it's missing, `live.sh` says "20 s of pure silence" while something is playing.
2. **Translation languages:** download **Japanese** and **Chinese (Simplified)** under **System Settings → General → Language & Region → Translation Languages**. `live.sh` tells you if they're missing.

## How it works

1. **Capture**: `live_captions_helper` (Swift, Core Audio process tap) records system audio at 16 kHz mono. It is audio-only, so DRM-protected video players keep showing their video
2. **Segment**: an energy gate cuts the stream at short pauses (≤ 6 s per chunk)
3. **Transcribe**: `mlx-whisper` with the same hallucination filters as the batch tool
4. **Translate**: Apple's on-device Translation framework turns it into Chinese: free, offline, no API key, about 0.05 s a line. With `--translator claude`, Claude (`claude-haiku-5-5`) translates instead: paid, but better with casual speech because it sees the previous lines
5. **Show**: the helper draws a click-through subtitle bar at the bottom of the active screen

```bash
./live.sh                    # Ctrl+C to stop
./live.sh --show-original    # Chinese + Japanese
./live.sh --translator claude   # Claude API instead of Apple (needs .env, see below)
./live.sh --font-size 42 --hold 8
./live.sh --file clip.mp4    # test with a file instead of system audio (needs ffmpeg)
```

| Flag | Default | Description |
|---|---|---|
| `--show-original` | off | Also show the transcribed line |
| `--font-size` / `--hold` | `34` / `6` | Subtitle size (pt) / seconds it stays up |
| `--capture` | `tap` | `screen` uses ScreenCaptureKit instead (DRM players then show a black screen) |
| `--translator` | `apple` | `apple` (on-device, free) or `claude` (Claude API, paid) |
| `--claude-model` | `claude-haiku-5-5` | Claude model used with `--translator claude` |
| `--lang` | `ja` | Spoken language code, or `auto` |
| `--silence-db` | `-45` | Raise it (e.g. `-35`) if background music keeps chunks from ending |
| `--pause` / `--max-len` | `0.5` / `6` | Pause that ends a line / longest chunk (s) |

**Claude translation (optional):** run `cp .env.example .env` and put your key from the [Anthropic Console](https://console.anthropic.com/) in `ANTHROPIC_API_KEY`. Claude Haiku 5.5 costs about $0.05 per 2-hour video. Anthropic's usage policy restricts sexually explicit content, so keep the default Apple translator for that kind of video.

## Subtitle files for videos (`run.sh`)

A separate, optional tool: transcribes Japanese audio from a video file with Whisper, translates each subtitle segment to Simplified Chinese via Claude, and writes a `.srt` subtitle file. Unlike live captions, it uses the paid Claude API.

### How it works

1. **Extract** — ffmpeg cuts the requested time range from the video into a temp wav file
2. **Transcribe** — `mlx-whisper` (Apple MLX backend) transcribes the audio in Japanese
3. **Clean** — drops Whisper hallucinations (e.g. `ご視聴ありがとうございました` over silence), empty segments and repetition loops
4. **Translate** — Claude (`claude-opus-4-6`) translates each segment to Simplified Chinese in batches
5. **Output** — writes a `.srt` file with correct timestamps matching the original video

### Requirements

Everything live captions need, plus:

- `ffmpeg`: `brew install ffmpeg`
- An Anthropic API key in `.env`

### Setup

```bash
cp .env.example .env   # then fill in your key
```

`.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Usage

```bash
# Basic — transcribe entire video with default model (large-v3-turbo)
./run.sh video.mp4

# Specify output file
./run.sh video.mp4 -o subtitles.srt

# Transcribe only the first 20 minutes
./run.sh video.mp4 --duration 20m

# Transcribe from 20min to 40min
./run.sh video.mp4 --duration 20m-40m

# Multiple ranges (comma-separated, or repeat --duration) — one combined .srt
./run.sh video.mp4 --duration "90-180,10m-12m30s,1:05:00-"
./run.sh video.mp4 --duration 1:30-3:00 --duration 6m-7m

# Use a different Whisper model
./run.sh video.mp4 -m large
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `<video>.srt` | Output `.srt` file path |
| `-m`, `--model` | `large-v3-turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `large-v3-turbo` |
| `--duration` | *(full video)* | One or more ranges, comma-separated or repeated: `20m`, `20m-40m`, `90-180`, `1:30-3:00`, `1h-` (to end). Times are bare seconds, `h`/`m`/`s` units, or `[hh:]mm:ss`. Overlapping ranges are merged. |
| `--batch-size` | `20` | Subtitle segments per Claude API call |

## Models

Whisper models are downloaded from HuggingFace (`mlx-community`) on first use and cached at `~/.cache/huggingface/hub/`.

| Model | Size | Notes |
|---|---|---|
| `large-v3-turbo` | ~1.6 GB | **Default.** ~3x faster than large, similar accuracy |
| `large` | ~3 GB | Maximum accuracy |
| `medium` | ~1.5 GB | Good balance |
| `small` / `base` / `tiny` | <500 MB | Fast, lower accuracy |

## Project structure

```
live.sh                     # checks prerequisites, builds the helper, starts live captions
live_captions.py            # live captions: segmenting and transcription
live_captions_helper.swift  # live captions: audio capture, Apple translation, subtitle overlay
run.sh                      # .srt tool wrapper (no venv activation needed)
transcribe_to_srt.py        # .srt tool; its model table and filters are shared with live captions
requirements.txt            # Python packages for both tools
.env.example                # template for the .srt tool's API key
.venv/                      # Python virtual environment (git-ignored)
```
