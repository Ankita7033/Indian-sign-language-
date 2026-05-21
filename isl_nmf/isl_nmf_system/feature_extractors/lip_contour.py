"""
feature_extractors/lip_contour.py
===================================
Extracts lip geometry features for ISL non-manual analysis:

  1. Mouth Aspect Ratio (MAR): vertical opening / horizontal width
  2. Lip spread ratio: how wide the lip corners are stretched
  3. Lip protrusion proxy: mean Z-depth of lip landmarks (relative to face)
  4. Lip contour area: convex hull of outer lip landmarks (normalised)
  5. Corner asymmetry: difference in corner Y positions

ISL relevance:
  - Mouthing (silent speech) accompanies many ISL signs
  - Lip rounding / spreading distinguishes certain classifiers
  - Mouth opening encodes question / negation markers
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.config import SystemConfig, DEFAULT_CONFIG, LandmarkIndices
from feature_extractors.face_landmarks import FaceLandmarkResult
from utils.math_utils import ema_filter, euclidean_distance
from utils.logger import get_logger

log = get_logger(__name__)
_LI = LandmarkIndices


def _contour_area_from_indices(pts: np.ndarray, indices) -> float:
    """Compute 2-D convex hull area of a set of landmark points."""
    sub = pts[indices, :2].astype(np.float32)
    hull = cv2.convexHull(sub)
    return float(cv2.contourArea(hull))


@dataclass
class LipContourFeatures:
    # Core geometry
    mar: float = 0.0               # Mouth Aspect Ratio
    lip_spread: float = 0.0        # corner_dist / face_width
    lip_open: float = 0.0          # vertical opening, normalised
    lip_protrusion: float = 0.0    # Z-depth proxy
    outer_lip_area: float = 0.0    # convex hull area, normalised by face_w^2
    corner_asymmetry: float = 0.0  # |left_corner_y - right_corner_y| / face_h

    # Discrete states
    mouth_open:   bool = False
    mouth_spread: bool = False
    lip_rounded:  bool = False     # low spread + mid opening = rounding proxy
    lip_pursed:   bool = False     # very low spread + some protrusion

    # For visualisation / downstream
    outer_lip_pts: Optional[np.ndarray] = None   # (N, 2) px
    inner_lip_pts: Optional[np.ndarray] = None


class LipContourExtractor:
    """
    Extracts lip shape features using MediaPipe FaceMesh lip landmark indices.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr   = config.thresholds
        self._alpha = self.thr.smoother_alpha

        # EMA state
        self._mar  = 0.0
        self._spr  = 0.0
        self._open = 0.0
        self._prot = 0.0
        
        self._base_spr = 0.0
        self._base_alpha = 0.01
        self._initialized = False
        log.info("LipContourExtractor ready.")

    def process(self, lm_result: FaceLandmarkResult) -> LipContourFeatures:
        feat = LipContourFeatures()
        if not lm_result.face_detected or lm_result.face_pts_px is None:
            return feat

        pts  = lm_result.face_pts_px
        fw   = max(lm_result.face_width, 1.0)
        fh   = max(lm_result.face_height, 1.0)

        # ---- Mouth Aspect Ratio (MAR) ----
        # Vertical: top to bottom of outer lip
        top_y    = pts[_LI.LIP_TOP,    1]
        bot_y    = pts[_LI.LIP_BOTTOM, 1]
        vert     = abs(bot_y - top_y)

        left_x   = pts[_LI.LIP_LEFT_CORNER,  0]
        right_x  = pts[_LI.LIP_RIGHT_CORNER, 0]
        horiz    = abs(right_x - left_x) + 1e-6

        mar_raw  = vert / horiz
        self._mar = ema_filter(self._mar, mar_raw, self._alpha)
        feat.mar = self._mar

        # ---- Lip open (normalised by face height) ----
        open_raw = vert / fh
        self._open = ema_filter(self._open, open_raw, self._alpha)
        feat.lip_open = self._open

        # ---- Lip spread ----
        spread_raw = horiz / fw
        
        if not self._initialized:
            self._spr = self._base_spr = spread_raw
            self._initialized = True
            
        self._base_spr = ema_filter(self._base_spr, spread_raw, self._base_alpha)
        
        # We only care about positive spread relative to resting
        calib_spread = max(0.0, spread_raw - self._base_spr)
        self._spr = ema_filter(self._spr, calib_spread, self._alpha)
        feat.lip_spread = self._spr

        # ---- Lip protrusion (Z proxy) ----
        # Mean Z of inner lip landmarks minus mean Z of nose bridge
        inner_ids = _LI.INNER_LIP_UPPER + _LI.INNER_LIP_LOWER
        inner_z = np.mean(pts[inner_ids, 2])
        nose_z  = pts[_LI.NOSE_BRIDGE, 2]
        prot_raw = float(nose_z - inner_z) / fw   # positive = forward
        self._prot = ema_filter(self._prot, prot_raw, self._alpha)
        feat.lip_protrusion = self._prot

        # ---- Outer lip area ----
        area_raw = _contour_area_from_indices(pts, _LI.OUTER_LIP_UPPER + _LI.OUTER_LIP_LOWER)
        feat.outer_lip_area = area_raw / (fw * fw)

        # ---- Corner asymmetry ----
        left_cy  = pts[_LI.LIP_LEFT_CORNER,  1]
        right_cy = pts[_LI.LIP_RIGHT_CORNER, 1]
        feat.corner_asymmetry = abs(left_cy - right_cy) / fh

        # ---- Point arrays for visualiser ----
        feat.outer_lip_pts = pts[_LI.OUTER_LIP_UPPER + _LI.OUTER_LIP_LOWER, :2]
        feat.inner_lip_pts = pts[_LI.INNER_LIP_UPPER + _LI.INNER_LIP_LOWER, :2]

        # ---- Discrete states ----
        feat.mouth_open   = feat.lip_open > self.thr.lip_open_threshold
        feat.mouth_spread = feat.lip_spread > self.thr.lip_spread_threshold
        feat.lip_rounded  = (feat.lip_spread < 0.45) and (feat.lip_open > 0.015)
        feat.lip_pursed   = (feat.lip_spread < 0.40) and (feat.lip_protrusion > self.thr.lip_protrusion_threshold)

        return feat
