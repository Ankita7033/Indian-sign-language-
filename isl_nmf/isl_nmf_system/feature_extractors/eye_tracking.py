"""
feature_extractors/eye_tracking.py
====================================
Extracts:
  1. Eye Aspect Ratio (EAR) for blink / wide-eye detection
  2. Iris-based gaze direction using MediaPipe iris landmarks
  3. Gaze state classification: center / left / right / up / down

EAR formula (Soukupová & Čech, 2016):
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  where p1..p6 are 6 eye contour landmarks in order.

Iris gaze: displacement of iris centre relative to eye bounding box
centre, normalised by eye box width/height.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.config import SystemConfig, DEFAULT_CONFIG, LandmarkIndices
from feature_extractors.face_landmarks import FaceLandmarkResult
from utils.math_utils import euclidean_distance, ema_filter
from utils.logger import get_logger

log = get_logger(__name__)
_LI = LandmarkIndices


def _compute_ear(pts: np.ndarray, ear_ids) -> float:
    """
    Compute Eye Aspect Ratio from 6 landmark indices.
    ear_ids = [p1, p2, p3, p4, p5, p6]
    """
    p = pts[ear_ids, :2]
    A = euclidean_distance(p[1], p[5])
    B = euclidean_distance(p[2], p[4])
    C = euclidean_distance(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-9)


@dataclass
class EyeTrackingFeatures:
    left_ear:  float = 0.0
    right_ear: float = 0.0
    mean_ear:  float = 0.0

    # Blink / wide states
    is_blinking:   bool = False
    is_wide_open:  bool = False
    left_wide:     bool = False
    right_wide:    bool = False

    # Gaze (normalised displacement: -1 left, +1 right / up/down)
    gaze_x: float = 0.0    # horizontal
    gaze_y: float = 0.0    # vertical

    # Gaze direction label
    gaze_direction: str = "center"   # center / left / right / up / down

    # Iris available?
    iris_available: bool = False


class EyeTracker:
    """
    Combines EAR blink detection with iris-landmark gaze estimation.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr    = config.thresholds
        self._alpha = self.thr.smoother_alpha

        # EMA state
        self._lear  = 0.3
        self._rear  = 0.3
        self._gx    = 0.0
        self._gy    = 0.0

        log.info("EyeTracker ready.")

    def _iris_gaze(self, pts: np.ndarray,
                   iris_center: np.ndarray,
                   inner_idx: int, outer_idx: int,
                   upper_ids, lower_ids) -> tuple:
        """
        Compute normalised iris displacement within eye bounding box.
        Returns (gaze_x, gaze_y) each in [-1, 1].
        """
        inner  = pts[inner_idx, :2]
        outer  = pts[outer_idx, :2]
        upper  = np.mean(pts[upper_ids, :2], axis=0)
        lower  = np.mean(pts[lower_ids, :2], axis=0)

        cx = (inner[0] + outer[0]) / 2.0
        cy = (upper[1] + lower[1]) / 2.0
        ew  = abs(outer[0] - inner[0]) + 1e-6
        eh  = abs(lower[1] - upper[1]) + 1e-6

        ic = iris_center[:2]
        gx = (ic[0] - cx) / ew
        gy = (ic[1] - cy) / eh
        return float(gx), float(gy)

    def process(self, lm_result: FaceLandmarkResult) -> EyeTrackingFeatures:
        feat = EyeTrackingFeatures()
        if not lm_result.face_detected or lm_result.face_pts_px is None:
            return feat

        pts = lm_result.face_pts_px

        # ---- EAR ----
        lear_raw = _compute_ear(pts, _LI.LEFT_EAR_LANDMARKS)
        rear_raw = _compute_ear(pts, _LI.RIGHT_EAR_LANDMARKS)
        self._lear = ema_filter(self._lear, lear_raw, self._alpha)
        self._rear = ema_filter(self._rear, rear_raw, self._alpha)

        feat.left_ear  = self._lear
        feat.right_ear = self._rear
        feat.mean_ear  = (self._lear + self._rear) / 2.0

        feat.is_blinking  = feat.mean_ear < self.thr.ear_blink_threshold
        feat.left_wide    = self._lear > self.thr.ear_wide_threshold
        feat.right_wide   = self._rear > self.thr.ear_wide_threshold
        feat.is_wide_open = feat.left_wide and feat.right_wide

        # ---- Iris gaze ----
        if lm_result.iris_left_px is not None and lm_result.iris_right_px is not None:
            feat.iris_available = True
            # Left iris center = index 0 of iris_left_px (landmark 468)
            lgx, lgy = self._iris_gaze(
                pts,
                lm_result.iris_left_px[0],
                inner_idx=362, outer_idx=263,
                upper_ids=_LI.LEFT_EYE_UPPER,
                lower_ids=_LI.LEFT_EYE_LOWER
            )
            rgx, rgy = self._iris_gaze(
                pts,
                lm_result.iris_right_px[0],
                inner_idx=133, outer_idx=33,
                upper_ids=_LI.RIGHT_EYE_UPPER,
                lower_ids=_LI.RIGHT_EYE_LOWER
            )
            raw_gx = (lgx + rgx) / 2.0
            raw_gy = (lgy + rgy) / 2.0
            self._gx = ema_filter(self._gx, raw_gx, self._alpha)
            self._gy = ema_filter(self._gy, raw_gy, self._alpha)
            feat.gaze_x = self._gx
            feat.gaze_y = self._gy

            # Direction classification
            ht = self.thr.gaze_horizontal_threshold
            vt = self.thr.gaze_vertical_threshold
            if abs(self._gx) > ht:
                feat.gaze_direction = "left" if self._gx < 0 else "right"
            elif abs(self._gy) > vt:
                feat.gaze_direction = "up" if self._gy < 0 else "down"
            else:
                feat.gaze_direction = "center"

        return feat
