"""
datasets/dataset_recorder.py
==============================
Dataset Recorder Mode — Feature #4

Press R → start recording a sample
Press S → save with annotation
Press ESC → stop recording

Stores per-sample:
  - Raw video frames (MP4)
  - MediaPipe landmark JSON per frame
  - Semantic token annotations
  - Confidence scores
  - Timestamp metadata

Output structure:
  datasets/recorded/
    sample_001/
      video.mp4
      landmarks.json
      annotation.json
    sample_002/
      ...

This makes your project a self-improving research platform.
Researchers can record their own ISL samples and build
a labelled dataset directly from the running system.
"""

import cv2
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

DATASET_DIR = "datasets/recorded"


@dataclass
class LandmarkFrame:
    frame_idx: int
    timestamp: float
    face_landmarks: List[List[float]]   # (N, 3) flattened
    pose_landmarks: List[List[float]]   # (M, 3) flattened
    feature_vector: Dict[str, float]
    semantic_tokens: List[str]
    confidence_scores: Dict[str, int]


@dataclass
class RecordedSample:
    sample_id: str
    start_time: str
    end_time: str
    fps: int
    n_frames: int
    annotated_tokens: List[str]
    annotator_note: str
    frames: List[LandmarkFrame] = field(default_factory=list)


class DatasetRecorder:
    """
    Records ISL signing samples with full landmark + token annotations.

    Keyboard controls (pass key to handle_key() each frame):
      R / r  → toggle recording ON/OFF
      S / s  → save current recording with annotation prompt
      1-9    → quick-annotate with preset token (during recording)
    """

    # Quick annotation presets
    QUICK_TOKENS = {
        ord('1'): ["QUESTION(type=WH)"],
        ord('2'): ["QUESTION(type=YN)"],
        ord('3'): ["NEGATION(active)"],
        ord('4'): ["EMPHASIS(strong)"],
        ord('5'): ["DOUBT"],
        ord('6'): ["AGREEMENT"],
        ord('7'): ["SURPRISE"],
        ord('8'): ["TOPIC_SHIFT(true)"],
        ord('9'): ["NEUTRAL"],
    }

    def __init__(self, output_dir: str = DATASET_DIR,
                 fps: int = 30,
                 frame_size: tuple = (640, 480)):
        self.output_dir  = output_dir
        self.fps         = fps
        self.frame_size  = frame_size

        self._recording  = False
        self._frames: List[LandmarkFrame] = []
        self._video_frames: List[np.ndarray] = []
        self._sample_id  = ""
        self._start_time = 0.0
        self._quick_annotation: List[str] = []

        os.makedirs(output_dir, exist_ok=True)
        log.info("DatasetRecorder ready. Output: %s", output_dir)

    @property
    def is_recording(self) -> bool:
        return self._recording

    def toggle_recording(self) -> str:
        """Start or stop recording. Returns status message."""
        if not self._recording:
            self._recording   = True
            self._frames      = []
            self._video_frames = []
            self._sample_id   = f"sample_{int(time.time())}"
            self._start_time  = time.time()
            self._quick_annotation = []
            msg = f"🔴 RECORDING STARTED — {self._sample_id}"
            log.info(msg)
        else:
            self._recording = False
            duration = time.time() - self._start_time
            msg = f"⏹ RECORDING STOPPED — {len(self._frames)} frames ({duration:.1f}s)"
            log.info(msg)
        return msg

    def feed_frame(self,
                   bgr_frame: np.ndarray,
                   frame_idx: int,
                   feature_vector: Dict[str, float],
                   semantic_tokens: List[str],
                   confidence_scores: Dict[str, int],
                   landmark_result=None) -> None:
        """Feed one frame to the recorder (only stores if recording)."""
        if not self._recording:
            return

        # Store resized video frame
        small = cv2.resize(bgr_frame, self.frame_size)
        self._video_frames.append(small)

        # Extract landmarks if available
        face_lms = []
        pose_lms = []
        if landmark_result and landmark_result.face_pts_px is not None:
            face_lms = landmark_result.face_pts_px.tolist()
        if landmark_result and landmark_result.pose_landmarks:
            plms = landmark_result.pose_landmarks
            # Tasks API: plain list; solutions API: object with .landmark
            lm_list = plms if isinstance(plms, list) else plms.landmark
            pose_lms = [[lm.x, lm.y, lm.z] for lm in lm_list]

        lm_frame = LandmarkFrame(
            frame_idx        = frame_idx,
            timestamp        = time.time(),
            face_landmarks   = face_lms,
            pose_landmarks   = pose_lms,
            feature_vector   = dict(feature_vector),
            semantic_tokens  = list(semantic_tokens),
            confidence_scores = dict(confidence_scores),
        )
        self._frames.append(lm_frame)

    def handle_key(self, key: int) -> Optional[str]:
        """
        Handle keyboard input during recording.
        Returns a status message or None.
        """
        if key in (ord('r'), ord('R')):
            return self.toggle_recording()
        if key in (ord('s'), ord('S')) and not self._recording:
            return self.save()
        if key in self.QUICK_TOKENS and self._recording:
            self._quick_annotation = self.QUICK_TOKENS[key]
            return f"📌 Quick annotation: {self._quick_annotation}"
        return None

    def save(self, note: str = "") -> str:
        """Save the current recording to disk."""
        if not self._frames:
            return "⚠ Nothing to save."

        sample_dir = os.path.join(self.output_dir, self._sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # ── Save video ──────────────────────────────────────────────
        video_path = os.path.join(sample_dir, "video.mp4")
        if self._video_frames:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w   = self._video_frames[0].shape[:2]
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            for vf in self._video_frames:
                writer.write(vf)
            writer.release()

        # ── Save landmarks ──────────────────────────────────────────
        lm_path = os.path.join(sample_dir, "landmarks.json")
        lm_data = [asdict(f) for f in self._frames]
        with open(lm_path, "w") as fp:
            json.dump(lm_data, fp, indent=2)

        # ── Collect token distribution ──────────────────────────────
        all_tokens: Dict[str, int] = {}
        for f in self._frames:
            for tok in f.semantic_tokens:
                all_tokens[tok] = all_tokens.get(tok, 0) + 1
        dominant = sorted(all_tokens, key=lambda x: -all_tokens[x])[:3]

        # ── Save annotation ─────────────────────────────────────────
        annotation = {
            "sample_id":        self._sample_id,
            "start_time":       time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(self._start_time)),
            "end_time":         time.strftime("%Y-%m-%d %H:%M:%S"),
            "fps":              self.fps,
            "n_frames":         len(self._frames),
            "duration_seconds": len(self._frames) / self.fps,
            "annotated_tokens": self._quick_annotation or dominant,
            "token_distribution": all_tokens,
            "annotator_note":   note,
        }
        ann_path = os.path.join(sample_dir, "annotation.json")
        with open(ann_path, "w") as fp:
            json.dump(annotation, fp, indent=2)

        msg = (f"✅ Saved: {self._sample_id} "
               f"({len(self._frames)} frames) → {sample_dir}")
        log.info(msg)

        # Reset
        self._frames = []
        self._video_frames = []
        self._sample_id = ""
        return msg

    def get_status_overlay(self) -> str:
        """Returns a one-line status for display on the video frame."""
        if self._recording:
            elapsed = time.time() - self._start_time
            n = len(self._frames)
            ann = self._quick_annotation[0].split("(")[0] if self._quick_annotation else "auto"
            return f"● REC {elapsed:.1f}s | {n} frames | ann: {ann}"
        return "[ R=Record  S=Save  1-9=QuickAnnotate ]"
