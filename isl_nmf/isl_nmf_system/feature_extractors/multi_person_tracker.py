"""
feature_extractors/multi_person_tracker.py
============================================
Multi-Person Detection Mode — Feature #8

Tracks multiple signers using face detection + tracking IDs.
Each tracked person gets their own feature extraction pipeline.

Example output:
  Person 1 → QUESTION(type=WH)
  Person 2 → NEUTRAL (listener)

Uses OpenCV face detection + simple centroid-based tracking
(no deep learning required).

Real-world use: classrooms, meetings, video calls.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrackedPerson:
    person_id: int
    bbox: Tuple[int,int,int,int]    # x, y, w, h
    centroid: Tuple[int,int]
    active_tokens: List[str] = field(default_factory=list)
    frames_tracked: int = 0
    last_seen: int = 0
    is_primary: bool = False        # largest/most centered face


class MultiPersonTracker:
    """
    Detects and tracks multiple faces using OpenCV Haar cascades.
    Assigns stable IDs using centroid distance matching.

    Real-time, no GPU needed, works alongside MediaPipe.
    """

    MAX_DISAPPEARED = 30   # frames before dropping a track
    MAX_DISTANCE    = 100  # pixels for ID matching

    def __init__(self):
        # Load OpenCV face detector
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._tracked: Dict[int, TrackedPerson] = {}
        self._next_id = 0
        self._frame_idx = 0
        log.info("MultiPersonTracker ready (Haar cascade).")

    def _detect_faces(self, gray: np.ndarray) -> List[Tuple]:
        """Detect faces in grayscale frame."""
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (60, 60),
            flags        = cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces) if len(faces) > 0 else []

    def _centroid(self, bbox: Tuple) -> Tuple[int,int]:
        x, y, w, h = bbox
        return (x + w//2, y + h//2)

    def _match_to_existing(self, centroid: Tuple[int,int]) -> Optional[int]:
        """Find closest existing track within MAX_DISTANCE."""
        best_id   = None
        best_dist = self.MAX_DISTANCE

        for pid, person in self._tracked.items():
            cx, cy = person.centroid
            dist = ((centroid[0]-cx)**2 + (centroid[1]-cy)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = pid

        return best_id

    def update(self, bgr_frame: np.ndarray,
               frame_idx: int) -> List[TrackedPerson]:
        """
        Detect faces and update tracks. Returns list of active persons.
        """
        self._frame_idx = frame_idx
        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        # Match detections to existing tracks
        matched_ids = set()
        for bbox in faces:
            centroid = self._centroid(bbox)
            pid = self._match_to_existing(centroid)

            if pid is not None:
                # Update existing
                p = self._tracked[pid]
                p.bbox       = tuple(bbox)
                p.centroid   = centroid
                p.last_seen  = frame_idx
                p.frames_tracked += 1
                matched_ids.add(pid)
            else:
                # New person
                new_id = self._next_id
                self._next_id += 1
                self._tracked[new_id] = TrackedPerson(
                    person_id     = new_id,
                    bbox          = tuple(bbox),
                    centroid      = centroid,
                    frames_tracked = 1,
                    last_seen     = frame_idx,
                )
                matched_ids.add(new_id)

        # Remove disappeared tracks
        disappeared = [
            pid for pid, p in self._tracked.items()
            if frame_idx - p.last_seen > self.MAX_DISAPPEARED
        ]
        for pid in disappeared:
            del self._tracked[pid]

        # Mark primary person (largest face area)
        if self._tracked:
            primary = max(self._tracked.values(),
                          key=lambda p: p.bbox[2] * p.bbox[3])
            for p in self._tracked.values():
                p.is_primary = (p.person_id == primary.person_id)

        return list(self._tracked.values())

    def assign_tokens(self, person_id: int,
                      tokens: List[str]) -> None:
        """Assign detected tokens to a specific tracked person."""
        if person_id in self._tracked:
            self._tracked[person_id].active_tokens = tokens

    def draw_tracks(self, canvas: np.ndarray,
                    persons: List[TrackedPerson]) -> np.ndarray:
        """Draw bounding boxes and IDs on canvas."""
        for p in persons:
            x, y, w, h = p.bbox
            col = (50, 220, 50) if p.is_primary else (200, 180, 50)
            cv2.rectangle(canvas, (x,y), (x+w, y+h), col, 2)
            label = f"P{p.person_id}"
            if p.active_tokens:
                short = p.active_tokens[0].split("(")[0]
                label += f": {short}"
            cv2.putText(canvas, label, (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        return canvas
