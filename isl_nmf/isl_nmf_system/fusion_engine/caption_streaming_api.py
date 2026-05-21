"""
fusion_engine/caption_streaming_api.py
========================================
Feature 5: Caption Streaming API

Provides a real-world usable caption streaming interface.
Outputs captions in multiple formats:
  - SRT (subtitle file format for video)
  - WebVTT (browser-compatible captions)
  - JSON stream (for API integration)
  - Plain text log

Usage:
  api = CaptionStreamingAPI()
  api.push(frame_idx, confirmed_tokens, sentence, confidence, timestamp_ms)
  api.export_srt("output.srt")
  api.export_vtt("output.vtt")
  api.get_json_stream()  # returns last N entries as JSON

Real-world deployment: this API makes captions available to
any downstream application — video players, chat interfaces,
screen readers, accessibility tools.
"""

import json, os, time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import deque

CAPTIONS_DIR = "captions"


@dataclass
class CaptionEntry:
    index: int
    frame_start: int
    frame_end: int
    timestamp_start_ms: float
    timestamp_end_ms: float
    tokens: List[str]
    sentence: str
    subtitle: str
    confidence_pct: int
    is_final: bool = False


def _ms_to_srt_time(ms: float) -> str:
    total_s = ms / 1000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = int(total_s % 60)
    ms_part = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"


def _ms_to_vtt_time(ms: float) -> str:
    return _ms_to_srt_time(ms).replace(",", ".")


class CaptionStreamingAPI:
    """
    Real-time caption streaming with multi-format export.
    Maintains a rolling buffer of caption entries.
    """

    BUFFER_SIZE = 500
    DEFAULT_DURATION_MS = 2000   # 2 seconds per caption

    def __init__(self, fps: int = 30,
                 output_dir: str = CAPTIONS_DIR):
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._all_entries: List[CaptionEntry] = []
        self._counter = 0
        self._last_sentence = ""
        self._session_start_ms = time.time() * 1000

    def push(self, frame_idx: int,
             tokens: List[str],
             sentence: str,
             subtitle: str,
             confidence_pct: int,
             fps: int = 30) -> Optional[CaptionEntry]:
        """
        Push a new caption frame. Only creates a new entry when
        the sentence changes (avoids duplicate captions).
        """
        if not sentence or sentence == "..." or sentence == self._last_sentence:
            return None

        ts_ms = self._session_start_ms + (frame_idx / fps) * 1000
        entry = CaptionEntry(
            index               = self._counter,
            frame_start         = frame_idx,
            frame_end           = frame_idx + int(fps * 2),
            timestamp_start_ms  = ts_ms,
            timestamp_end_ms    = ts_ms + self.DEFAULT_DURATION_MS,
            tokens              = list(tokens),
            sentence            = sentence,
            subtitle            = subtitle,
            confidence_pct      = confidence_pct,
        )
        self._buffer.append(entry)
        self._all_entries.append(entry)
        self._last_sentence = sentence
        self._counter += 1
        return entry

    def get_json_stream(self, last_n: int = 10) -> str:
        """Returns last N entries as JSON — for API integration."""
        entries = list(self._buffer)[-last_n:]
        return json.dumps([asdict(e) for e in entries], indent=2)

    def export_srt(self, path: str = None) -> str:
        """Export all captions as SRT subtitle file."""
        if not path:
            path = os.path.join(
                self.output_dir,
                f"captions_{int(time.time())}.srt"
            )
        lines = []
        for e in self._all_entries:
            lines.append(str(e.index + 1))
            lines.append(f"{_ms_to_srt_time(e.timestamp_start_ms)} --> "
                         f"{_ms_to_srt_time(e.timestamp_end_ms)}")
            lines.append(e.sentence)
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"SRT exported: {path}")
        return path

    def export_vtt(self, path: str = None) -> str:
        """Export all captions as WebVTT file (browser-compatible)."""
        if not path:
            path = os.path.join(
                self.output_dir,
                f"captions_{int(time.time())}.vtt"
            )
        lines = ["WEBVTT", ""]
        for e in self._all_entries:
            lines.append(f"{_ms_to_vtt_time(e.timestamp_start_ms)} --> "
                         f"{_ms_to_vtt_time(e.timestamp_end_ms)}")
            lines.append(f"{e.sentence}  [{e.confidence_pct}%]")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"WebVTT exported: {path}")
        return path

    def export_json(self, path: str = None) -> str:
        if not path:
            path = os.path.join(
                self.output_dir,
                f"captions_{int(time.time())}.json"
            )
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._all_entries], f, indent=2)
        print(f"JSON exported: {path}")
        return path

    def get_live_caption(self) -> str:
        """Returns the most recent caption for live display."""
        if self._buffer:
            return self._buffer[-1].sentence
        return ""

    @property
    def total_captions(self) -> int:
        return self._counter
