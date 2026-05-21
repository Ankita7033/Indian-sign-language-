"""
feature_extractors/optical_flow_tracker.py
============================================
Computes dense optical flow using Gunnar-Farnebäck algorithm
across pre-defined facial ROIs to capture holistic motion fields
that complement point-landmark tracking.

Returns per-ROI motion magnitude, dominant direction, and flow maps
which contribute to the semantic fusion graph's motion channel.

ROIs are defined in normalised [0,1] coordinates (config.OPTICAL_FLOW_ROIS)
and re-computed to pixel coordinates each frame.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from config.config import SystemConfig, DEFAULT_CONFIG, OPTICAL_FLOW_ROIS
from utils.math_utils import ema_filter
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FlowROIResult:
    magnitude: float = 0.0        # mean optical flow magnitude (px/frame)
    direction_deg: float = 0.0    # dominant motion direction (0-360)
    dx: float = 0.0               # mean horizontal flow
    dy: float = 0.0               # mean vertical flow
    active: bool = False          # exceeds threshold?


@dataclass
class OpticalFlowFeatures:
    roi_results: Dict[str, FlowROIResult] = field(default_factory=dict)
    global_magnitude: float = 0.0
    global_dx: float = 0.0
    global_dy: float = 0.0
    flow_map: Optional[np.ndarray] = None   # (H,W,2) for visualiser


class OpticalFlowTracker:
    """
    Dense optical flow over configurable facial ROIs.

    Uses Farnebäck parameters tuned for real-time performance
    at 720p resolution.
    """

    # Farnebäck parameters
    FB_PARAMS = dict(
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 15,
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0
    )

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr   = config.thresholds
        self._alpha = self.thr.smoother_alpha

        self._prev_gray: Optional[np.ndarray] = None
        self._rois = OPTICAL_FLOW_ROIS

        # Per-ROI EMA state
        self._mag_ema: Dict[str, float] = {k: 0.0 for k in self._rois}

        log.info("OpticalFlowTracker ready (%d ROIs).", len(self._rois))

    def _get_roi_px(self, roi_norm: Tuple[float,float,float,float],
                    h: int, w: int) -> Tuple[int,int,int,int]:
        """Convert normalised ROI (x,y,w_,h_) to pixel coords."""
        x, y, rw, rh = roi_norm
        x1 = int(x * w);  y1 = int(y * h)
        x2 = int((x+rw) * w); y2 = int((y+rh) * h)
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    def process(self, bgr_frame: np.ndarray) -> OpticalFlowFeatures:
        feat = OpticalFlowFeatures()
        h, w = bgr_frame.shape[:2]

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            feat.roi_results = {k: FlowROIResult() for k in self._rois}
            return feat

        # Compute dense flow on full frame
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **self.FB_PARAMS
        )  # (H, W, 2)
        self._prev_gray = gray.copy()
        feat.flow_map = flow

        # Global stats
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        feat.global_magnitude = float(np.mean(mag))
        feat.global_dx = float(np.mean(flow[..., 0]))
        feat.global_dy = float(np.mean(flow[..., 1]))

        # Per-ROI analysis
        for roi_name, roi_norm in self._rois.items():
            x1, y1, x2, y2 = self._get_roi_px(roi_norm, h, w)
            roi_flow = flow[y1:y2, x1:x2]

            if roi_flow.size == 0:
                feat.roi_results[roi_name] = FlowROIResult()
                continue

            roi_mag, roi_ang = cv2.cartToPolar(
                roi_flow[..., 0], roi_flow[..., 1]
            )
            mean_mag = float(np.mean(roi_mag))

            # EMA smoothing per ROI
            self._mag_ema[roi_name] = ema_filter(
                self._mag_ema[roi_name], mean_mag, self._alpha
            )

            dom_ang = float(np.degrees(np.mean(roi_ang)))
            dx = float(np.mean(roi_flow[..., 0]))
            dy = float(np.mean(roi_flow[..., 1]))

            res = FlowROIResult(
                magnitude    = self._mag_ema[roi_name],
                direction_deg = dom_ang,
                dx = dx,
                dy = dy,
                active = mean_mag > self.thr.flow_magnitude_threshold
            )
            feat.roi_results[roi_name] = res

        return feat
