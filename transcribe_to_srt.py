#!/usr/bin/env python3
"""
Japanese video transcription and Chinese subtitle generator.

Transcribes Japanese audio from a video using OpenAI Whisper,
translates segments to Simplified Chinese with Claude, and writes a .srt file.

Requirements:
    pip install openai-whisper anthropic
    brew install ffmpeg  # or: apt install ffmpeg

Usage:
    python transcribe_to_srt.py video.mp4
    python transcribe_to_srt.py video.mp4 -o subtitles.srt -m large
"""

import re
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

# ── SRT helpers ──────────────────────────────────────────────────────────────

def fmt_ts(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], translations: list[str], output_path: str) -> None:
    """Write a .srt file from Whisper segments and translated lines."""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (seg, text) in enumerate(zip(segments, translations), start=1):
            f.write(f"{idx}\n")
            f.write(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n{text.strip()}\n\n")
    print(f"  Saved → {output_path}")


# ── Duration helpers ─────────────────────────────────────────────────────────

def _parse_duration(s: str) -> float:
    """Parse a time like '20m', '1h30m', '90s', '90', '1:30' or '1:02:03.5' into seconds.

    A bare number is seconds; colon form is [hh:]mm:ss.
    """
    s = s.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"(?:\d+:)?\d+:\d+(?:\.\d+)?", s):
        total = 0.0
        for part in s.split(":"):
            total = total * 60 + float(part)
        return total
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if not s or not m or not any(m.groups()):
        raise argparse.ArgumentTypeError(
            f"Invalid time '{s}'. Use formats like '90', '90s', '20m', '1h30m', '1:30'."
        )
    h, m_, s_ = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + m_ * 60 + s_


def parse_duration_range(value: str) -> tuple[float, float | None]:
    """Parse one range into (start_sec, end_sec | None).

    '20m'       → (0, 1200)
    '20m-40m'   → (1200, 2400)
    '1:30-2:45' → (90, 165)
    '1h-'       → (3600, None)   # to end of video
    """
    if '-' in value:
        start_str, end_str = value.split('-', 1)
        start = _parse_duration(start_str)
        if not end_str.strip():
            return start, None
        end = _parse_duration(end_str)
        if end <= start:
            raise argparse.ArgumentTypeError(f"End time must be after start time in '{value}'.")
        return start, end
    return 0.0, _parse_duration(value)


def parse_duration_ranges(values: list[str]) -> list[tuple[float, float | None]]:
    """Parse one or more comma-separated ranges; sort them and merge overlaps."""
    ranges = sorted(
        (parse_duration_range(part) for value in values for part in value.split(",") if part.strip()),
        key=lambda r: r[0],
    )
    merged: list[tuple[float, float | None]] = []
    for start, end in ranges:
        if merged:
            prev_start, prev_end = merged[-1]
            if prev_end is None or start <= prev_end:
                new_end = None if prev_end is None or end is None else max(prev_end, end)
                merged[-1] = (prev_start, new_end)
                continue
        merged.append((start, end))
    return merged


def _fmt_range(start: float, end: float | None) -> str:
    return f"{fmt_ts(start)[:-4]} – {fmt_ts(end)[:-4] if end is not None else 'end'}"


def extract_segment(video_path: str, start: float, end: float | None) -> tuple[str, tempfile.NamedTemporaryFile]:
    """Use ffmpeg to cut [start, end] from the video into a temp wav file.

    Returns (temp_path, temp_file_handle) — caller must close the handle when done.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", video_path]
    if end is not None:
        cmd += ["-t", str(end - start)]
    cmd += ["-ac", "1", "-ar", "16000", "-vn", tmp.name]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


# ── Transcription ─────────────────────────────────────────────────────────────

_MLX_MODELS = {
    "tiny":           "mlx-community/whisper-tiny-mlx",
    "base":           "mlx-community/whisper-base-mlx",
    "small":          "mlx-community/whisper-small-mlx",
    "medium":         "mlx-community/whisper-medium-mlx",
    "large":          "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# Whisper hallucinates on silence/music/non-speech (moaning, breathing) and,
# with condition_on_previous_text=True, feeds each hallucination into the next
# 30s window as a prompt — one bad window poisons the rest of the file.
_DECODE_OPTIONS = dict(
    condition_on_previous_text=False,     # stop hallucinations cascading
    word_timestamps=True,                 # required by hallucination_silence_threshold
    hallucination_silence_threshold=2.0,  # skip text Whisper invents over silent gaps
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
    compression_ratio_threshold=2.4,
)

# Stock phrases Whisper emits over silence (learned from YouTube-style captions).
_HALLUCINATION_PATTERNS = re.compile(
    r"ご視聴ありがとうございました|ご覧いただきありがとうございます|チャンネル登録"
    r"|^(字幕|翻訳|提供)[:：]"
)
_MIN_AVG_LOGPROB = -1.5  # windows below this are almost always invented text


def _collapse_repeats(text: str, max_repeats: int = 3) -> str:
    """Shorten runs like 'あ、あ、あ、あ…' or '下の下の下の…' to at most max_repeats units."""
    for unit_len in range(1, 9):
        pattern = re.compile(r"(.{%d}[、。,!?！？…\s]*)\1{%d,}" % (unit_len, max_repeats))
        text = pattern.sub(lambda m: m.group(1) * max_repeats, text)
    return text.strip()


def clean_segments(segments: list[dict]) -> list[dict]:
    """Drop hallucinated/empty segments and collapse repetition loops."""
    cleaned: list[dict] = []
    for seg in segments:
        text = _collapse_repeats(seg["text"].strip())
        if not text or seg["end"] - seg["start"] < 0.05:
            continue
        if _HALLUCINATION_PATTERNS.search(text):
            continue
        if seg.get("avg_logprob", 0.0) < _MIN_AVG_LOGPROB:
            continue
        prev = cleaned[-1] if cleaned else None
        # Merge back-to-back identical lines (a looping decoder) into one subtitle
        if prev and prev["text"] == text and seg["start"] - prev["end"] < 3.0:
            prev["end"] = seg["end"]
            continue
        cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})
    return cleaned


def transcribe(video_path: str, model_size: str,
               start_sec: float = 0.0, end_sec: float | None = None) -> list[dict]:
    """Return Whisper segments [{start, end, text}, ...] for the video.

    If start_sec/end_sec are given, only that range is transcribed and the
    returned timestamps are offset so they match the original video timeline.
    """
    import time
    import os

    audio_path = video_path
    tmp_path = None
    if start_sec > 0 or end_sec is not None:
        range_str = f"{start_sec}s–{end_sec}s" if end_sec else f"{start_sec}s–end"
        print(f"[1/3] Extracting segment {range_str} with ffmpeg …")
        tmp_path = extract_segment(video_path, start_sec, end_sec)
        audio_path = tmp_path

    try:
        import mlx_whisper
        repo = _MLX_MODELS[model_size]
        print(f"[1/3] Transcribing with mlx-whisper '{model_size}' (Apple MLX) …")
        t0 = time.time()
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=repo,
            language="ja",
            verbose=False,
            **_DECODE_OPTIONS,
        )
    except ImportError:
        import torch
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[1/3] Loading Whisper '{model_size}' model on {device} …")
        model = whisper.load_model(model_size, device=device)
        t0 = time.time()
        print(f"[1/3] Transcribing (language=ja) …")
        result = model.transcribe(audio_path, language="ja", task="transcribe",
                                  verbose=False, **_DECODE_OPTIONS)
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    segments = result["segments"]

    # Shift timestamps back to original video timeline
    if start_sec > 0:
        for seg in segments:
            seg["start"] += start_sec
            seg["end"]   += start_sec

    elapsed = time.time() - t0
    audio_duration = (segments[-1]["end"] - segments[0]["start"]) if segments else 0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    print(f"      {len(segments)} segments found — {elapsed:.1f}s for {audio_duration:.1f}s audio (RTF {rtf:.2f}x)")
    return segments


# ── Translation ───────────────────────────────────────────────────────────────

def _parse_numbered_lines(text: str, expected: int) -> list[str]:
    """Extract numbered translations from Claude's response."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept "1. 你好" or "1) 你好" or bare lines
        if line[0].isdigit():
            dot = line.find(". ")
            paren = line.find(") ")
            sep = min(x for x in (dot, paren) if x != -1) if (dot != -1 or paren != -1) else -1
            lines.append(line[sep + 2:].strip() if sep != -1 else line)
        else:
            lines.append(line)

    if len(lines) == expected:
        return lines

    # Fallback: return what we got, pad if short
    print(f"      ⚠  Expected {expected} lines, parsed {len(lines)} — padding if needed.")
    while len(lines) < expected:
        lines.append("")
    return lines[:expected]


def translate_segments(segments: list[dict], batch_size: int) -> list[str]:
    """Translate all Japanese segments to Simplified Chinese via Claude."""
    client = anthropic.Anthropic()
    translations: list[str] = []
    total = len(segments)
    batches = (total + batch_size - 1) // batch_size

    print(f"[2/3] Translating {total} segments in {batches} batch(es) …")

    for b_idx in range(batches):
        batch = segments[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"      Batch {b_idx + 1}/{batches} ({len(batch)} segments) …", end=" ", flush=True)

        numbered = "\n".join(
            f"{i + 1}. {seg['text'].strip()}" for i, seg in enumerate(batch)
        )

        prompt = (
            "Translate each numbered Japanese subtitle line into Simplified Chinese.\n"
            "Rules:\n"
            "- Return ONLY the numbered list, one translation per line.\n"
            "- Keep the same numbering format: '1. 译文'\n"
            "- Keep subtitles concise and natural.\n"
            "- Do NOT add explanations or extra text.\n\n"
            f"{numbered}"
        )

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},  # translation is straightforward
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            final = stream.get_final_message()

        # Extract the text block (skip any thinking blocks)
        response_text = next(
            (block.text for block in final.content if block.type == "text"), ""
        )

        batch_translations = _parse_numbered_lines(response_text, len(batch))
        translations.extend(batch_translations)
        print("✓")

    return translations


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a Japanese video and generate a Chinese .srt subtitle file."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "-o", "--output",
        help="Output .srt path (default: <video>.srt)",
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3-turbo",
        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"],
        help="Whisper model size (default: large-v3-turbo; ~3x faster than large with similar accuracy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Subtitle segments per Claude API call (default: 50)",
    )
    parser.add_argument(
        "--duration",
        metavar="RANGE",
        default=None,
        action="append",
        help="Time range(s) to transcribe. '20m' = first 20 min, '20m-40m' = 20–40 min, "
             "'1h-' = 1h to end. Separate multiple ranges with commas or repeat the flag, "
             "e.g. '90-180,10m-12m30s' or '1:30-3:00'. Times: bare seconds, h/m/s units, or [hh:]mm:ss.",
    )
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        sys.exit(f"Error: file not found — {video_path}")

    output_path = args.output or str(Path(video_path).with_suffix(".srt"))

    ranges: list[tuple[float, float | None]] = [(0.0, None)]
    if args.duration:
        try:
            ranges = parse_duration_ranges(args.duration)
        except argparse.ArgumentTypeError as e:
            sys.exit(f"Error: {e}")

    print(f"\nInput : {video_path}")
    print(f"Output: {output_path}")
    if args.duration:
        for i, (start, end) in enumerate(ranges, start=1):
            print(f"Range {i}: {_fmt_range(start, end)}")
    print()

    # 1. Transcribe each range; timestamps stay on the original video timeline
    segments: list[dict] = []
    for i, (start, end) in enumerate(ranges, start=1):
        if len(ranges) > 1:
            print(f"── Range {i}/{len(ranges)}: {_fmt_range(start, end)}")
        raw = transcribe(video_path, args.model, start, end)
        kept = clean_segments(raw)
        print(f"      {len(kept)} segments kept after removing {len(raw) - len(kept)} hallucinated/duplicate ones")
        segments.extend(kept)
    if not segments:
        sys.exit("Error: no speech detected in video.")

    # 2. Translate
    translations = translate_segments(segments, args.batch_size)

    # 3. Write .srt
    print("[3/3] Writing .srt file …")
    write_srt(segments, translations, output_path)

    print(f"\nDone — {len(segments)} subtitles written to {output_path}")


if __name__ == "__main__":
    main()
