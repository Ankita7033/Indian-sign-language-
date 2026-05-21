"""
datasets/auto_annotation_engine.py
=====================================
Feature 2: Dataset Auto-Annotation Engine

Automatically generates ISLRTC-compatible annotation JSON
from the live system output without human labelling.

Workflow:
  1. Record video with --auto-annotate flag
  2. System assigns token labels per frame using confirmed outputs
  3. High-confidence frames (>80%) are marked as auto-annotated
  4. Low-confidence frames are flagged for human review
  5. Output: datasets/auto_annotated/session_TIMESTAMP.json

This creates a self-growing dataset from real ISL signing,
which can be used to validate and improve the system over time.

Research contribution: demonstrates dataset creation capability,
which is a standard requirement in NLP/CV papers.
"""

import json, os, time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict
from config.config import LinguisticTokens

T = LinguisticTokens
AUTO_ANNOT_DIR = "datasets/auto_annotated"
MIN_CONF_FOR_AUTO = 0.75     # threshold for auto-accept
MIN_SEGMENT_FRAMES = 5       # minimum frames for a valid segment


@dataclass
class AnnotatedSegment:
    frame_start: int
    frame_end:   int
    tokens: List[str]
    confidence: float
    auto_accepted: bool
    needs_review: bool
    method: str = "auto"


@dataclass
class AutoAnnotationSession:
    session_id: str
    start_time: str
    end_time: str = ""
    fps: int = 30
    total_frames: int = 0
    auto_accepted: int = 0
    needs_review: int = 0
    segments: List[AnnotatedSegment] = field(default_factory=list)


class AutoAnnotationEngine:
    """
    Converts live system output into structured annotation data.
    Groups consecutive frames with the same token set into segments.
    """

    def __init__(self, fps: int = 30,
                 output_dir: str = AUTO_ANNOT_DIR):
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._session = AutoAnnotationSession(
            session_id = f"session_{int(time.time())}",
            start_time = time.strftime("%Y-%m-%d %H:%M:%S"),
            fps        = fps,
        )
        # Current running segment
        self._seg_start: int = 0
        self._seg_tokens: List[str] = []
        self._seg_confs: List[float] = []
        self._frame_count: int = 0

    def feed(self, frame_idx: int,
             tokens: List[str],
             mean_confidence: float) -> None:
        """Feed one frame's output to the annotator."""
        self._frame_count += 1
        clean_tokens = [t for t in tokens if t != T.NEUTRAL]

        # Check if tokens changed — close current segment
        if clean_tokens != self._seg_tokens and self._seg_tokens:
            self._close_segment(frame_idx - 1)
            self._seg_start  = frame_idx
            self._seg_tokens = clean_tokens
            self._seg_confs  = [mean_confidence]
        elif not self._seg_tokens:
            self._seg_start  = frame_idx
            self._seg_tokens = clean_tokens
            self._seg_confs  = [mean_confidence]
        else:
            self._seg_confs.append(mean_confidence)

    def _close_segment(self, frame_end: int) -> None:
        duration = frame_end - self._seg_start + 1
        if duration < MIN_SEGMENT_FRAMES or not self._seg_tokens:
            return

        import numpy as np
        mean_conf = float(np.mean(self._seg_confs))
        auto_ok   = mean_conf >= MIN_CONF_FOR_AUTO

        seg = AnnotatedSegment(
            frame_start    = self._seg_start,
            frame_end      = frame_end,
            tokens         = list(self._seg_tokens),
            confidence     = mean_conf,
            auto_accepted  = auto_ok,
            needs_review   = not auto_ok,
        )
        self._session.segments.append(seg)
        if auto_ok:
            self._session.auto_accepted += 1
        else:
            self._session.needs_review += 1

    def save(self) -> str:
        """Close final segment and save session to JSON."""
        if self._seg_tokens:
            self._close_segment(self._frame_count)

        self._session.end_time    = time.strftime("%Y-%m-%d %H:%M:%S")
        self._session.total_frames = self._frame_count

        path = os.path.join(
            self.output_dir,
            f"{self._session.session_id}.json"
        )
        data = asdict(self._session)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        summary = (
            f"Auto-annotation saved: {path}\n"
            f"  Segments    : {len(self._session.segments)}\n"
            f"  Auto-accept : {self._session.auto_accepted}\n"
            f"  Needs review: {self._session.needs_review}"
        )
        print(summary)
        return path

    def get_stats(self) -> str:
        return (f"AutoAnnot: {len(self._session.segments)} segs | "
                f"accept={self._session.auto_accepted} "
                f"review={self._session.needs_review}")
