# Japanese Video Transcriber

Transcribes Japanese audio from video files using Whisper, translates each subtitle segment to Simplified Chinese via Claude, and outputs a `.srt` subtitle file.

## How it works

1. **Extract** — ffmpeg cuts the requested time range from the video into a temp wav file
2. **Transcribe** — `mlx-whisper` (Apple MLX backend) transcribes the audio in Japanese
3. **Clean** — drops Whisper hallucinations (e.g. `ご視聴ありがとうございました` over silence), empty segments and repetition loops
4. **Translate** — Claude (`claude-opus-4-6`) translates each segment to Simplified Chinese in batches
5. **Output** — writes a `.srt` file with correct timestamps matching the original video

## Requirements

- macOS with Apple Silicon (M-series) — uses `mlx-whisper` for fast on-device inference
- `ffmpeg` — installed via `brew install ffmpeg`
- Python 3.11+
- Anthropic API key in `.env`

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install openai-whisper anthropic python-dotenv mlx-whisper

cp .env.example .env   # then fill in your key
```

`.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

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

## Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `<video>.srt` | Output `.srt` file path |
| `-m`, `--model` | `large-v3-turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `large-v3-turbo` |
| `--duration` | *(full video)* | One or more ranges, comma-separated or repeated: `20m`, `20m-40m`, `90-180`, `1:30-3:00`, `1h-` (to end). Times are bare seconds, `h`/`m`/`s` units, or `[hh:]mm:ss`. Overlapping ranges are merged. |
| `--batch-size` | `20` | Subtitle segments per Claude API call |

## Live captions

`./live.sh` puts real-time Chinese subtitles over whatever the Mac is playing (any app, full-screen video included), with nothing saved to disk.

1. **Capture**: `live_captions_helper` (Swift, Core Audio process tap) records system audio at 16 kHz mono. It is audio-only, so DRM-protected video players keep showing their video
2. **Segment**: an energy gate cuts the stream at short pauses (≤ 6 s per chunk)
3. **Transcribe**: `mlx-whisper` with the same hallucination filters as the batch tool
4. **Translate**: Apple's on-device Translation framework turns it into Chinese: free, offline, about 0.05 s a line. `--translator claude` uses Claude (`claude-haiku-5-5`, paid API) instead, which handles casual speech better because it sees the previous lines
5. **Show**: the helper draws a click-through subtitle bar at the bottom of the active screen

```bash
./live.sh                    # Ctrl+C to stop
./live.sh --show-original    # Chinese + Japanese
./live.sh --font-size 42 --hold 8
./live.sh --file clip.mp4    # test with a file instead of system audio
```

Apple Translation needs the Japanese and Chinese (Simplified) languages downloaded under **System Settings → General → Language & Region → Translation Languages**; the helper tells you if they're missing. On first run, allow your terminal app under **System Settings → Privacy & Security → Screen & System Audio Recording**, then restart the terminal. The helper is compiled automatically (needs Xcode Command Line Tools).

| Flag | Default | Description |
|---|---|---|
| `--show-original` | off | Also show the transcribed line |
| `--font-size` / `--hold` | `34` / `6` | Subtitle size (pt) / seconds it stays up |
| `--capture` | `tap` | `screen` uses ScreenCaptureKit instead (DRM players then show a black screen) |
| `--lang` | `ja` | Spoken language code, or `auto` |
| `--translator` | `apple` | `apple` (on-device, free) or `claude` (API, paid) |
| `--claude-model` | `claude-haiku-5-5` | Claude model used with `--translator claude` |
| `--silence-db` | `-45` | Raise it (e.g. `-35`) if background music keeps chunks from ending |
| `--pause` / `--max-len` | `0.5` / `6` | Pause that ends a line / longest chunk (s) |

## Models

Whisper models are downloaded from HuggingFace (`mlx-community`) on first use and cached at `~/.cache/huggingface/hub/`.

| Model | Size | Notes |
|---|---|---|
| `large-v3-turbo` | ~800 MB | **Default.** ~3x faster than large, similar accuracy |
| `large` | ~3 GB | Maximum accuracy |
| `medium` | ~1.5 GB | Good balance |
| `small` / `base` / `tiny` | <500 MB | Fast, lower accuracy |

## Project structure

```
transcribe_to_srt.py   # main script
run.sh                 # convenience wrapper (no venv activation needed)
live_captions.py       # live captions: segmenting, transcription, translation
live_captions_helper.swift  # live captions: system audio capture + subtitle overlay
live.sh                # builds the helper and starts live captions
.env                   # API key (git-ignored)
.gitignore
.venv/                 # Python virtual environment (git-ignored)
```
