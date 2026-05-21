"""
utils/adaptive_threshold_engine.py
=====================================
Feature 4: Adaptive Threshold Engine

Dynamically adjusts detection thresholds per-user in real time
based on observed feature distributions during the session.

Unlike static calibration (run once), the adaptive engine
continuously monitors the baseline and updates thresholds
using a rolling window approach.

Method:
  For each feature channel, maintain a running estimate of:
    μ (mean) and σ (std) during NEUTRAL periods
  Threshold = μ + N_sigma × σ   (N_sigma tunable, default 2.5)

This means:
  - User with naturally high brows → threshold auto-raises
  - User in dim lighting → EAR threshold auto-adjusts
  - User tilts camera → head pose baseline auto-corrects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
from config.config import SystemConfig, DEFAULT_CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

N_SIGMA = 2.5          # standard deviations above baseline
MIN_SAMPLES = 60       # frames before adaptation kicks in
WINDOW = 300           # rolling window size (10 seconds at 30fps)
ADAPT_RATE = 0.05      # how fast thresholds update per frame


@dataclass
class ChannelStats:
    name: str
    buffer: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    mean: float = 0.0
    std:  float = 0.01
    threshold: float = 0.0
    n_samples: int = 0
    adapted: bool = False


class AdaptiveThresholdEngine:
    """
    Real-time per-user threshold adaptation.

    Feed neutral-face feature vectors to adapt().
    Read adapted thresholds from get_thresholds().
    Apply to config with apply_to_config().
    """

    CHANNELS = {
        "left_brow_raise":          ("eyebrow_raise_threshold",  0.048),
        "right_brow_raise":         ("eyebrow_raise_threshold",  0.048),
        "mean_ear":                 ("ear_wide_threshold",       0.36),
        "head_pitch":               ("head_nod_threshold",       22.0),
        "head_yaw":                 ("head_shake_threshold",     25.0),
        "head_roll":                ("head_tilt_threshold",      18.0),
        "lip_open":                 ("lip_open_threshold",       0.060),
        "shoulder_bilateral_raise": ("shoulder_raise_threshold", 0.070),
    }

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG,
                 n_sigma: float = N_SIGMA):
        self.cfg     = config
        self.n_sigma = n_sigma
        self._stats: Dict[str, ChannelStats] = {
            ch: ChannelStats(name=ch, threshold=default)
            for ch, (_, default) in self.CHANNELS.items()
        }
        self._frame_count = 0
        self._is_neutral_fn = None  # optional: callable(fv) -> bool
        log.info("AdaptiveThresholdEngine ready.")

    def feed(self, feature_vector: Dict[str, float],
             is_neutral: bool = True) -> None:
        """
        Feed one frame. Only updates baseline during neutral periods.
        """
        self._frame_count += 1
        if not is_neutral:
            return

        for ch, stats in self._stats.items():
            val = abs(float(feature_vector.get(ch, 0.0)))
            stats.buffer.append(val)
            stats.n_samples += 1

            if stats.n_samples >= MIN_SAMPLES:
                arr = np.array(stats.buffer)
                new_mean = float(np.mean(arr))
                new_std  = float(np.std(arr)) + 0.001

                # Smooth update
                stats.mean = (1 - ADAPT_RATE) * stats.mean + ADAPT_RATE * new_mean
                stats.std  = (1 - ADAPT_RATE) * stats.std  + ADAPT_RATE * new_std

                # Compute new threshold
                stats.threshold = stats.mean + self.n_sigma * stats.std
                stats.adapted = True

    def apply_to_config(self) -> int:
        """
        Apply adapted thresholds to the system config.
        Returns number of thresholds updated.
        """
        thr = self.cfg.thresholds
        updated = 0

        # Eyebrow — take max of left/right
        bl = self._stats.get("left_brow_raise")
        br = self._stats.get("right_brow_raise")
        if bl and br and bl.adapted and br.adapted:
            thr.eyebrow_raise_threshold = max(bl.threshold, br.threshold)
            updated += 1

        # Head pose
        for ch, (config_attr, _) in self.CHANNELS.items():
            stats = self._stats.get(ch)
            if stats and stats.adapted and ch not in ("left_brow_raise", "right_brow_raise"):
                setattr(thr, config_attr, stats.threshold)
                updated += 1

        if updated:
            log.debug("Adaptive thresholds updated: %d channels", updated)
        return updated

    def get_status(self) -> str:
        adapted = sum(1 for s in self._stats.values() if s.adapted)
        total   = len(self._stats)
        return f"Adaptive: {adapted}/{total} channels adapted | frame={self._frame_count}"

    def get_thresholds_summary(self) -> Dict[str, float]:
        return {ch: s.threshold for ch, s in self._stats.items() if s.adapted}
