"""
utils/user_calibration.py
===========================
Personalized Calibration Mode — Feature #3

Different people sign differently. Their resting brow height,
head tilt, and lip movement baseline all vary significantly.

Run:  python main.py --webcam --calibrate-user

The system records 10 seconds of neutral face data and computes:
  - Neutral eyebrow height baseline
  - Neutral head pitch/yaw/roll baseline
  - Neutral eye openness (EAR) baseline
  - Neutral shoulder position baseline

These baselines are saved to: config/user_profile.json
On next run they are loaded and used to OFFSET all feature
measurements, making detection relative to that specific user.

This is essential for accessibility tools — one threshold
does not fit all signers.
"""

import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

from utils.logger import get_logger

log = get_logger(__name__)

PROFILE_PATH = "config/user_profile.json"
CALIBRATION_SECONDS = 10
TARGET_FPS = 30
CALIBRATION_FRAMES = CALIBRATION_SECONDS * TARGET_FPS


@dataclass
class UserProfile:
    """Stores per-user neutral baseline measurements."""
    user_id: str = "default"
    calibrated: bool = False
    calibration_date: str = ""

    # Eyebrow baselines
    brow_height_left:  float = 0.048   # default threshold
    brow_height_right: float = 0.048
    brow_height_std:   float = 0.005

    # Head pose baselines
    head_pitch_neutral: float = 0.0
    head_yaw_neutral:   float = 0.0
    head_roll_neutral:  float = 0.0
    head_pose_std:      float = 2.0

    # Eye baselines
    ear_neutral: float = 0.30
    ear_std:     float = 0.02

    # Shoulder baselines (set by ShoulderTracker internally)
    shoulder_calibrated: bool = False

    # Lip baselines
    mar_neutral:   float = 0.08
    mar_std:       float = 0.01

    # Derived thresholds (computed from baselines + N*std)
    eyebrow_raise_threshold: float = 0.072
    head_nod_threshold:      float = 22.0
    head_shake_threshold:    float = 25.0
    ear_wide_threshold:      float = 0.40
    lip_open_threshold:      float = 0.060


def load_profile(path: str = PROFILE_PATH) -> UserProfile:
    """Load user profile from JSON, return defaults if not found."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            profile = UserProfile(**data)
            log.info("User profile loaded: %s (calibrated: %s)",
                     profile.user_id, profile.calibrated)
            return profile
        except Exception as e:
            log.warning("Could not load profile: %s. Using defaults.", e)
    return UserProfile()


def save_profile(profile: UserProfile, path: str = PROFILE_PATH) -> None:
    """Save user profile to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    log.info("User profile saved to %s", path)


class UserCalibrator:
    """
    Runs the calibration sequence and computes personalized thresholds.

    Usage:
      calibrator = UserCalibrator()
      # In webcam loop for CALIBRATION_FRAMES:
      calibrator.feed_frame(feature_vector)
      # After loop:
      profile = calibrator.compute_profile()
      save_profile(profile)
    """

    def __init__(self):
        self._buffers: Dict[str, List[float]] = {
            "brow_h_l":  [],
            "brow_h_r":  [],
            "head_pitch":[],
            "head_yaw":  [],
            "head_roll": [],
            "ear":       [],
            "mar":       [],
        }
        self._frame_count = 0
        self._start_time  = time.time()

    def feed_frame(self, feature_vector: Dict[str, float],
                   head_pitch: float = 0.0,
                   head_yaw: float   = 0.0,
                   head_roll: float  = 0.0,
                   ear: float        = 0.30,
                   mar: float        = 0.08) -> float:
        """
        Feed one frame's features. Returns calibration progress 0-1.
        """
        fv = feature_vector
        self._buffers["brow_h_l"].append(fv.get("left_brow_raise", 0.0))
        self._buffers["brow_h_r"].append(fv.get("right_brow_raise", 0.0))
        self._buffers["head_pitch"].append(head_pitch)
        self._buffers["head_yaw"].append(head_yaw)
        self._buffers["head_roll"].append(head_roll)
        self._buffers["ear"].append(ear)
        self._buffers["mar"].append(mar)
        self._frame_count += 1
        return min(self._frame_count / CALIBRATION_FRAMES, 1.0)

    def compute_profile(self, user_id: str = "default") -> UserProfile:
        """
        Compute personalized thresholds from calibration data.
        Thresholds = mean + 2.5 * std (so only deliberate actions fire).
        """
        def stats(key):
            arr = np.array(self._buffers[key])
            return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.01

        bl_m, bl_s = stats("brow_h_l")
        br_m, br_s = stats("brow_h_r")
        hp_m, hp_s = stats("head_pitch")
        hy_m, hy_s = stats("head_yaw")
        hr_m, hr_s = stats("head_roll")
        ear_m, ear_s = stats("ear")
        mar_m, mar_s = stats("mar")

        N = 2.5   # number of standard deviations above baseline

        profile = UserProfile(
            user_id             = user_id,
            calibrated          = True,
            calibration_date    = time.strftime("%Y-%m-%d %H:%M:%S"),
            brow_height_left    = bl_m,
            brow_height_right   = br_m,
            brow_height_std     = (bl_s + br_s) / 2,
            head_pitch_neutral  = hp_m,
            head_yaw_neutral    = hy_m,
            head_roll_neutral   = hr_m,
            head_pose_std       = max(hp_s, hy_s, hr_s, 1.0),
            ear_neutral         = ear_m,
            ear_std             = ear_s,
            mar_neutral         = mar_m,
            mar_std             = mar_s,

            # Personalised thresholds
            eyebrow_raise_threshold = bl_m + N * max(bl_s, 0.005),
            head_nod_threshold      = abs(hp_m) + N * max(hp_s, 2.0),
            head_shake_threshold    = abs(hy_m) + N * max(hy_s, 2.0),
            ear_wide_threshold      = ear_m + N * max(ear_s, 0.01),
            lip_open_threshold      = mar_m + N * max(mar_s, 0.005),
        )

        log.info("Calibration complete for user '%s'", user_id)
        log.info("  Brow threshold   : %.4f", profile.eyebrow_raise_threshold)
        log.info("  Head nod thr     : %.1f deg", profile.head_nod_threshold)
        log.info("  Head shake thr   : %.1f deg", profile.head_shake_threshold)

        return profile


def apply_profile_to_config(profile: UserProfile, config) -> None:
    """Apply a loaded UserProfile's thresholds to SystemConfig."""
    if not profile.calibrated:
        return
    thr = config.thresholds
    thr.eyebrow_raise_threshold = profile.eyebrow_raise_threshold
    thr.head_nod_threshold      = profile.head_nod_threshold
    thr.head_shake_threshold    = profile.head_shake_threshold
    thr.ear_wide_threshold      = profile.ear_wide_threshold
    thr.lip_open_threshold      = profile.lip_open_threshold
    log.info("Personalized thresholds applied from user profile.")
