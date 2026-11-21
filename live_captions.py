#!/usr/bin/env python3
"""
Real-time Chinese subtitles for whatever your Mac is playing.

System audio (Core Audio process tap, via live_captions_helper) → speech chunks cut
at pauses → mlx-whisper (Japanese) → Simplified Chinese → a floating subtitle bar
drawn by the helper above every window, full-screen video included.

Translation is Apple's on-device Translation framework, done inside the helper:
free, offline, no API key.

Usage:
    ./live.sh                         # caption system audio
    ./live.sh --show-original         # also show the Japanese line
    ./live.sh --file clip.mp4         # test on a file instead of system audio
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

# Reuse the batch tool's model table and hallucination filters.
from transcribe_to_srt import _MLX_MODELS, clean_segments

SAMPLE_RATE = 16000
FRAME = 480                      # 30 ms analysis frames
FRAME_SEC = FRAME / SAMPLE_RATE
HELPER = Path(__file__).parent / ".build" / "live_captions_helper"


# ── Audio sources ────────────────────────────────────────────────────────────

def read_pcm(stream, frames: queue.Queue) -> None:
    """Split a raw float32 PCM byte stream into FRAME-sized numpy arrays."""
    nbytes = FRAME * 4
    buf = b""
    while chunk := stream.read1(65536):
        buf += chunk
        while len(buf) >= nbytes:
            frames.put(np.frombuffer(buf[:nbytes], dtype=np.float32))
            buf = buf[nbytes:]
    frames.put(None)  # end of stream


def file_source(path: str, frames: queue.Queue) -> None:
    """Decode a media file with ffmpeg and feed it at real-time speed (for testing)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-re", "-i", path, "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
    )
    read_pcm(proc.stdout, frames)


# ── Segmentation ─────────────────────────────────────────────────────────────

class Segmenter:
    """Cut the stream into utterances: start on sound, end on a short pause or max length.

    Deliberately just an energy gate, not a VAD model — Silero misses whispered
    and non-standard speech in this content (see transcribe_to_srt.py history).
    """

    def __init__(self, silence_db: float, pause: float, max_len: float):
        self.silence_db = silence_db
        self.pause_frames = int(pause / FRAME_SEC)
        self.max_frames = int(max_len / FRAME_SEC)
        self.preroll: deque = deque(maxlen=int(0.3 / FRAME_SEC))
        self.frames: list[np.ndarray] = []
        self.voiced = 0
        self.quiet = 0

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame; returns a finished utterance when one ends."""
        rms = float(np.sqrt(np.mean(frame * frame)))
        loud = 20 * np.log10(rms + 1e-10) > self.silence_db

        if not self.frames:
            self.preroll.append(frame)
            if loud:
                self.frames = list(self.preroll)
                self.voiced, self.quiet = 1, 0
            return None

        self.frames.append(frame)
        if loud:
            self.voiced += 1
            self.quiet = 0
        else:
            self.quiet += 1
        if self.quiet >= self.pause_frames or len(self.frames) >= self.max_frames:
            return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        frames, voiced = self.frames, self.voiced
        self.frames, self.voiced, self.quiet = [], 0, 0
        self.preroll.clear()
        if voiced * FRAME_SEC < 0.25:  # a click or a blip, not speech
            return None
        return np.concatenate(frames)


# ── Transcription ────────────────────────────────────────────────────────────

# Short-chunk version of transcribe_to_srt._DECODE_OPTIONS: no previous-text
# conditioning (stops hallucination cascades); word timestamps aren't worth
# their latency on a few seconds of audio.
_DECODE_OPTIONS = dict(
    condition_on_previous_text=False,
    temperature=(0.0, 0.2, 0.4),
    compression_ratio_threshold=2.4,
)


def transcribe_chunk(audio: np.ndarray, repo: str, language: str | None) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=language,
                                    verbose=None, **_DECODE_OPTIONS)
    return " ".join(seg["text"] for seg in clean_segments(result["segments"]))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live Chinese subtitles for your Mac's audio.")
    parser.add_argument("-m", "--model", default="large-v3-turbo", choices=list(_MLX_MODELS),
                        help="Whisper model (default: large-v3-turbo)")
    parser.add_argument("--lang", default="ja",
                        help="Spoken language code, or 'auto' to detect (default: ja)")
    parser.add_argument("--show-original", action="store_true",
                        help="Show the transcribed line under the Chinese one")
    parser.add_argument("--font-size", type=float, default=34, help="Subtitle size in points (default: 34)")
    parser.add_argument("--hold", type=float, default=6, help="Seconds a subtitle stays up (default: 6)")
    parser.add_argument("--silence-db", type=float, default=-45,
                        help="Level below which audio counts as silence, dBFS (default: -45)")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Pause that ends a line, seconds (default: 0.5)")
    parser.add_argument("--max-len", type=float, default=6,
                        help="Longest chunk before a forced cut, seconds (default: 6)")
    parser.add_argument("--capture", default="tap", choices=["tap", "screen"],
                        help="Audio source: 'tap' records audio only (default); 'screen' uses "
                             "ScreenCaptureKit, which makes DRM-protected players go black")
    parser.add_argument("--file", help="Caption this media file instead of system audio (testing)")
    args = parser.parse_args()

    if not HELPER.exists():
        sys.exit(f"Error: {HELPER} not built — run ./live.sh, which compiles it.")

    helper_cmd = [str(HELPER), "--font-size", str(args.font_size), "--hold", str(args.hold),
                  "--capture", "none" if args.file else args.capture, "--source", args.lang, "--translate"]
    if args.show_original:
        helper_cmd.append("--show-original")
    helper = subprocess.Popen(helper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send(msg: dict) -> None:
        try:
            helper.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            helper.stdin.flush()
        except BrokenPipeError:
            pass  # helper exited; main() reports why once its audio stream ends

    def translate(ja: str) -> None:
        print(f"{time.strftime('%H:%M:%S')}  {ja}", flush=True)  # the helper prints the Chinese
        send({"ja": ja})

    repo = _MLX_MODELS[args.model]
    language = None if args.lang == "auto" else args.lang
    print(f"Loading Whisper '{args.model}' …", flush=True)
    transcribe_chunk(np.zeros(SAMPLE_RATE, dtype=np.float32), repo, language)  # warm-up
    print(f"{time.strftime('%H:%M:%S')}  字幕已就绪", flush=True)
    send({"zh": "字幕已就绪"})  # has "zh", so shown as-is

    frames: queue.Queue = queue.Queue()
    if args.file:
        threading.Thread(target=file_source, args=(args.file, frames), daemon=True).start()
    else:
        threading.Thread(target=read_pcm, args=(helper.stdout, frames), daemon=True).start()

    segmenter = Segmenter(args.silence_db, args.pause, args.max_len)
    print("Listening — Ctrl+C to stop.\n", flush=True)
    try:
        while (frame := frames.get()) is not None:
            if (audio := segmenter.push(frame)) is not None and (text := transcribe_chunk(audio, repo, language)):
                translate(text)
        if (audio := segmenter.flush()) is not None and (text := transcribe_chunk(audio, repo, language)):
            translate(text)
        time.sleep(args.hold if args.file else 0)  # let the last subtitle be seen
    except KeyboardInterrupt:
        pass
    finally:
        helper.terminate()
        try:
            helper.stdin.close()
        except BrokenPipeError:
            pass
    if helper.wait() not in (0, -15) and not args.file:
        sys.exit("Error: audio capture stopped — see the helper's message above.")


if __name__ == "__main__":
    main()
