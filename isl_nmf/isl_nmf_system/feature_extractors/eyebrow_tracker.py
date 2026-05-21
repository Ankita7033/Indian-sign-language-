"""
feature_extractors/eyebrow_tracker.py
======================================
Extracts eyebrow motion features from MediaPipe FaceMesh landmarks:
  - Eyebrow raise (bilateral / unilateral)
  - Eyebrow furrow / knit
  - Asymmetry index (left vs right)
  - Temporal velocity of brow movement

All distances are normalised by inter-pupillary distance (IPD) so
the system is robust to camera distance variation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from config.config import (
    SystemConfig, DEFAULT_CONFIG, LandmarkIndices, ThresholdConfig
)
from feature_extractors.face_landmarks import FaceLandmarkResult
from utils.math_utils import ema_filter, centroid
from utils.logger import get_logger

log = get_logger(__name__)

_LI = LandmarkIndices


@dataclass
class EyebrowFeatures:
    # Normalised height of each brow above eye
    left_brow_height:  float = 0.0
    right_brow_height: float = 0.0
    mean_brow_height:  float = 0.0

    # Normalised inter-brow horizontal distance (furrow proxy)
    interbrow_distance: float = 0.0

    # Derived booleans
    left_raised:  bool = False
    right_raised: bool = False
    both_raised:  bool = False
    furrowed:     bool = False
    asymmetric:   bool = False

    # Asymmetry score (0 = symmetric, 1 = maximally asymmetric)
    asymmetry_index: float = 0.0

    # Frame-to-frame velocity (normalised)
    brow_velocity: float = 0.0


class EyebrowTracker:
    """
    Eyebrow motion extractor using MediaPipe FaceMesh 468-point model.

    Height is measured as the vertical gap between the lower eyebrow
    contour centre and the upper eyelid centre, normalised by IPD.

    Inter-brow distance is the horizontal gap between the medial ends
    of both brows, also normalised by IPD.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr   = config.thresholds
        self._alpha = self.thr.smoother_alpha

        # EMA state
        self._lh = 0.0   # left height
        self._rh = 0.0   # right height
        self._ib = 0.0   # interbrow
        self._prev_mean = 0.0

        log.info("EyebrowTracker ready.")

    def _brow_height(self, pts: np.ndarray,
                     upper_ids: List[int], lower_ids: List[int],
                     eye_upper_ids: List[int]) -> float:
        """
        Returns normalised vertical gap between brow lower edge
        and eye upper edge.
        """
        brow_lower_y  = np.mean(pts[lower_ids, 1])
        eye_upper_y   = np.mean(pts[eye_upper_ids, 1])
        # In image coords, Y increases downward; brow is ABOVE eye
        # so brow_y < eye_y → height = eye_y - brow_y
        return float(eye_upper_y - brow_lower_y)

    def process(self, lm_result: FaceLandmarkResult) -> EyebrowFeatures:
        feat = EyebrowFeatures()
        if not lm_result.face_detected or lm_result.face_pts_px is None:
            return feat

        pts = lm_result.face_pts_px        # (N, 3) pixel coords
        ipd = lm_result.ipd_px

        # ---- Left brow height ----
        lh_raw = self._brow_height(
            pts,
            _LI.LEFT_EYEBROW_UPPER, _LI.LEFT_EYEBROW_LOWER,
            _LI.LEFT_EYE_UPPER
        )
        self._lh = ema_filter(self._lh, lh_raw / ipd, self._alpha)

        # ---- Right brow height ----
        rh_raw = self._brow_height(
            pts,
            _LI.RIGHT_EYEBROW_UPPER, _LI.RIGHT_EYEBROW_LOWER,
            _LI.RIGHT_EYE_UPPER
        )
        self._rh = ema_filter(self._rh, rh_raw / ipd, self._alpha)

        # ---- Inter-brow distance (furrow) ----
        # Medial end of left brow = index 336 (inner left)
        # Medial end of right brow = index 107 (inner right)
        ib_raw = abs(pts[336, 0] - pts[107, 0]) / ipd
        self._ib = ema_filter(self._ib, ib_raw, self._alpha)

        feat.left_brow_height   = self._lh
        feat.right_brow_height  = self._rh
        feat.mean_brow_height   = (self._lh + self._rh) / 2.0
        feat.interbrow_distance = self._ib

        # ---- Temporal velocity ----
        feat.brow_velocity = abs(feat.mean_brow_height - self._prev_mean)
        self._prev_mean = feat.mean_brow_height

        # ---- Classify raise ----
        thr = self.thr.eyebrow_raise_threshold
        feat.left_raised  = self._lh > thr
        feat.right_raised = self._rh > thr
        feat.both_raised  = feat.left_raised and feat.right_raised

        # ---- Classify furrow ----
        feat.furrowed = self._ib < self.thr.eyebrow_furrow_threshold

        # ---- Asymmetry ----
        feat.asymmetry_index = abs(self._lh - self._rh)
        feat.asymmetric = feat.asymmetry_index > self.thr.eyebrow_asymmetry_threshold

        return feat
