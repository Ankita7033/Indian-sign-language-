"""
feature_extractors/head_pose.py
================================
Estimates 3-D head pose (pitch, yaw, roll) from 6 stable facial
landmarks using OpenCV solvePnP (Levenberg-Marquardt).

Outputs
-------
HeadPoseFeatures dataclass:
    pitch_deg  : float   (+ = nodding down)
    yaw_deg    : float   (+ = turning right)
    roll_deg   : float   (+ = tilting right)
    is_nodding : bool
    is_shaking : bool
    is_tilting : bool
    nod_direction   : str  ("down" | "up" | "neutral")
    shake_direction : str  ("right" | "left" | "neutral")
    tilt_direction  : str  ("right" | "left" | "neutral")
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from config.config import (
    SystemConfig, DEFAULT_CONFIG,
    LandmarkIndices, FACE_3D_MODEL_POINTS, ThresholdConfig
)
from feature_extractors.face_landmarks import FaceLandmarkResult
from utils.math_utils import ema_filter
from utils.logger import get_logger

log = get_logger(__name__)

# Pre-cast 3D model points
_MODEL_3D = np.array(FACE_3D_MODEL_POINTS, dtype=np.float64)
_LM_IDS   = LandmarkIndices.POSE_LANDMARK_IDS


@dataclass
class HeadPoseFeatures:
    pitch_deg: float = 0.0
    yaw_deg:   float = 0.0
    roll_deg:  float = 0.0

    is_nodding:   bool = False
    is_shaking:   bool = False
    is_tilting:   bool = False

    nod_direction:   str = "neutral"
    shake_direction: str = "neutral"
    tilt_direction:  str = "neutral"

    rotation_vector:    Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    valid: bool = False


class HeadPoseEstimator:
    """
    Computes head pose via solvePnP on 6 stable face landmarks.

    The camera intrinsic matrix is estimated from frame dimensions
    assuming a standard pinhole model (no distortion).
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.thr = config.thresholds
        self._alpha = self.thr.smoother_alpha

        # Smoothed angles (EMA)
        self._pitch = 0.0
        self._yaw   = 0.0
        self._roll  = 0.0
        
        self._base_alpha = 0.01
        self._base_pitch = 0.0
        self._base_yaw   = 0.0
        self._base_roll  = 0.0
        self._initialized = False

        # Temporal pitch buffer for nod detection
        self._pitch_history: List[float] = []
        self._yaw_history:   List[float] = []
        self._roll_history:  List[float] = []
        self._hist_len = self.thr.smoother_window

        log.info("HeadPoseEstimator ready.")

    def _build_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        focal = img_w
        cx, cy = img_w / 2.0, img_h / 2.0
        return np.array([
            [focal,   0.0, cx],
            [  0.0, focal, cy],
            [  0.0,   0.0,  1.0]
        ], dtype=np.float64)

    def process(self, lm_result: FaceLandmarkResult) -> HeadPoseFeatures:
        feat = HeadPoseFeatures()
        if not lm_result.face_detected or lm_result.face_pts_px is None:
            return feat

        pts = lm_result.face_pts_px
        img_w, img_h = lm_result.img_w, lm_result.img_h

        # 2-D image points of the 6 reference landmarks
        image_pts_2d = np.array(
            [[pts[idx][0], pts[idx][1]] for idx in _LM_IDS],
            dtype=np.float64
        )

        cam_matrix = self._build_camera_matrix(img_w, img_h)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            _MODEL_3D, image_pts_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return feat

        # Convert rotation vector to Euler angles
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack([rot_mat, trans_vec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

        euler_flat = euler.flatten()
        raw_pitch = float(euler_flat[0])
        raw_yaw   = float(euler_flat[1])
        raw_roll  = float(euler_flat[2])

        # Instant initialization for baseline to prevent startup lag
        if not self._initialized:
            self._pitch = self._base_pitch = raw_pitch
            self._yaw   = self._base_yaw   = raw_yaw
            self._roll  = self._base_roll  = raw_roll
            self._initialized = True

        # EMA smoothing for instantaneous pose
        self._pitch = ema_filter(self._pitch, raw_pitch, self._alpha)
        self._yaw   = ema_filter(self._yaw,   raw_yaw,   self._alpha)
        self._roll  = ema_filter(self._roll,  raw_roll,  self._alpha)

        # Very slow EMA for baseline
        self._base_pitch = ema_filter(self._base_pitch, raw_pitch, self._base_alpha)
        self._base_yaw   = ema_filter(self._base_yaw,   raw_yaw,   self._base_alpha)
        self._base_roll  = ema_filter(self._base_roll,  raw_roll,  self._base_alpha)

        feat.pitch_deg = self._pitch - self._base_pitch
        feat.yaw_deg   = self._yaw - self._base_yaw
        feat.roll_deg  = self._roll - self._base_roll
        feat.rotation_vector    = rot_vec
        feat.translation_vector = trans_vec
        feat.valid = True

        # ---- Classify nod ----
        if abs(feat.pitch_deg) > self.thr.head_nod_threshold:
            feat.is_nodding = True
            feat.nod_direction = "down" if feat.pitch_deg > 0 else "up"

        # ---- Classify shake ----
        if abs(feat.yaw_deg) > self.thr.head_shake_threshold:
            feat.is_shaking = True
            feat.shake_direction = "right" if feat.yaw_deg > 0 else "left"

        # ---- Classify tilt ----
        if abs(feat.roll_deg) > self.thr.head_tilt_threshold:
            feat.is_tilting = True
            feat.tilt_direction = "right" if feat.roll_deg > 0 else "left"

        return feat
