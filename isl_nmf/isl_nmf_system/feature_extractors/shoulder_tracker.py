"""
feature_extractors/shoulder_tracker.py
========================================
Extracts shoulder posture features from MediaPipe Pose landmarks.

Features extracted:
  - Bilateral raise: both shoulders elevated relative to baseline
  - Lateral lean: body tilts left or right
  - Unilateral shrug: one shoulder markedly higher than the other
  - Shoulder roll / forward lean proxy (via Z-depth of shoulder landmarks)

ISL relevance:
  Shoulder shrug encodes doubt / uncertainty in ISL.
  Forward/backward lean marks emphasis and topic shifts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from config.config import SystemConfig, DEFAULT_CONFIG, PoseLandmarkIndices
from feature_extractors.face_landmarks import FaceLandmarkResult
from utils.math_utils import ema_filter
from utils.logger import get_logger

log = get_logger(__name__)
_PI = PoseLandmarkIndices


@dataclass
class ShoulderFeatures:
    # Normalised shoulder Y positions (0 = top of frame)
    left_shoulder_y:  float = 0.0
    right_shoulder_y: float = 0.0

    # Bilateral raise: mean shoulder Y relative to calibration baseline
    bilateral_raise:  float = 0.0   # positive = shoulders raised

    # Lateral lean: signed, + = lean right
    lateral_lean:     float = 0.0

    # Unilateral shrug: |left_y - right_y| (normalised)
    unilateral_shrug: float = 0.0
    shrug_side: str = "none"         # "left" | "right" | "none"

    # Z-depth proxy for forward lean (shoulder Z, normalised)
    forward_lean:     float = 0.0

    # Discrete states
    is_shrugging:  bool = False
    is_leaning:    bool = False
    is_raised:     bool = False


class ShoulderTracker:
    """
    Uses MediaPipe Pose shoulder / ear / hip landmarks to compute
    shoulder posture features.

    Calibration: the first CALIB_FRAMES frames are used to establish
    a neutral baseline for bilateral raise detection.
    """

    CALIB_FRAMES = 30

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr   = config.thresholds
        self._alpha = self.thr.smoother_alpha

        # EMA state
        self._ly   = 0.5
        self._ry   = 0.5
        self._bl   = 0.0
        self._lean = 0.0
        self._fwd  = 0.0

        # Calibration
        self._calib_count = 0
        self._baseline_ly = None
        self._baseline_ry = None
        self._calib_buf_l = []
        self._calib_buf_r = []

        log.info("ShoulderTracker ready (calibrating for %d frames).",
                 self.CALIB_FRAMES)

    def _get_pose_pt(self, pose_lms, idx: int, img_w: int, img_h: int) -> np.ndarray:
        # Tasks API returns a plain list; solutions API returns an object with .landmark
        lm = pose_lms[idx] if isinstance(pose_lms, list) else pose_lms.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])

    def process(self, lm_result: FaceLandmarkResult) -> ShoulderFeatures:
        feat = ShoulderFeatures()
        if not lm_result.pose_detected or lm_result.pose_landmarks is None:
            return feat

        plms = lm_result.pose_landmarks
        w, h = lm_result.img_w, lm_result.img_h

        # Retrieve key landmarks
        ls  = self._get_pose_pt(plms, _PI.LEFT_SHOULDER,  w, h)
        rs  = self._get_pose_pt(plms, _PI.RIGHT_SHOULDER, w, h)
        le  = self._get_pose_pt(plms, _PI.LEFT_EAR,       w, h)
        re  = self._get_pose_pt(plms, _PI.RIGHT_EAR,      w, h)
        lhp = self._get_pose_pt(plms, _PI.LEFT_HIP,       w, h)
        rhp = self._get_pose_pt(plms, _PI.RIGHT_HIP,      w, h)

        # Normalise Y by image height
        ly_norm = ls[1] / h
        ry_norm = rs[1] / h

        # EMA smoothing
        self._ly   = ema_filter(self._ly,   ly_norm, self._alpha)
        self._ry   = ema_filter(self._ry,   ry_norm, self._alpha)

        feat.left_shoulder_y  = self._ly
        feat.right_shoulder_y = self._ry

        # ---- Calibration phase ----
        if self._calib_count < self.CALIB_FRAMES:
            self._calib_buf_l.append(ly_norm)
            self._calib_buf_r.append(ry_norm)
            self._calib_count += 1
            if self._calib_count == self.CALIB_FRAMES:
                self._baseline_ly = float(np.mean(self._calib_buf_l))
                self._baseline_ry = float(np.mean(self._calib_buf_r))
                log.info("Shoulder baseline calibrated: L=%.3f R=%.3f",
                         self._baseline_ly, self._baseline_ry)
            return feat

        # ---- Bilateral raise ----
        # Smaller Y = higher in frame = raised shoulder
        delta_l = self._baseline_ly - self._ly
        delta_r = self._baseline_ry - self._ry
        self._bl = ema_filter(self._bl, (delta_l + delta_r) / 2.0, self._alpha)
        feat.bilateral_raise = self._bl

        # ---- Lateral lean ----
        # Hip midpoint Y compared to shoulder midpoint Y tilt
        hip_mid_x  = (lhp[0] + rhp[0]) / 2.0
        sho_mid_x  = (ls[0]  + rs[0])  / 2.0
        lean_raw   = (sho_mid_x - hip_mid_x) / w  # signed
        self._lean = ema_filter(self._lean, lean_raw, self._alpha)
        feat.lateral_lean = self._lean

        # ---- Unilateral shrug ----
        feat.unilateral_shrug = abs(self._ly - self._ry)
        if feat.unilateral_shrug > self.thr.shoulder_raise_threshold:
            feat.shrug_side = "left" if self._ly < self._ry else "right"

        # ---- Forward lean proxy ----
        # Z axis in MediaPipe Pose: more negative = closer to camera
        fwd_raw = -(ls[2] + rs[2]) / (2.0 * w)
        self._fwd = ema_filter(self._fwd, fwd_raw, self._alpha)
        feat.forward_lean = self._fwd

        # ---- Discrete states ----
        feat.is_shrugging = (feat.bilateral_raise > self.thr.shoulder_shrug_threshold or
                             feat.unilateral_shrug > self.thr.shoulder_raise_threshold)
        feat.is_leaning   = abs(feat.lateral_lean) > self.thr.shoulder_lean_threshold
        feat.is_raised    = feat.bilateral_raise > self.thr.shoulder_raise_threshold

        return feat
