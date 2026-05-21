"""
feature_extractors/temporal_smoother.py
=========================================
Temporal smoothing and gesture-boundary detection module.

Provides:
  1. SignalBuffer  — ring-buffer with moving-average and EMA access
  2. GestureSegmenter — detects onset/offset of non-manual gestures
     using a hysteresis threshold model
  3. TemporalSmoother — wraps all per-channel buffers and exposes
     a unified smoothed feature dict

Used AFTER per-frame feature extraction to reduce jitter and
provide temporal context to the semantic fusion graph.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque

from config.config import SystemConfig, DEFAULT_CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class SignalBuffer:
    """
    Fixed-length ring buffer for a scalar signal.
    Provides moving average and exponential moving average.
    """

    def __init__(self, window: int = 7, alpha: float = 0.35):
        self._window = window
        self._alpha  = alpha
        self._buf: Deque[float] = deque(maxlen=window)
        self._ema: float = 0.0
        self._initialized = False

    def update(self, val: float) -> None:
        if not self._initialized:
            self._ema = val
            self._initialized = True
        else:
            self._ema = self._alpha * val + (1 - self._alpha) * self._ema
        self._buf.append(val)

    @property
    def moving_average(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean(self._buf))

    @property
    def ema(self) -> float:
        return self._ema

    @property
    def std(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        return float(np.std(self._buf))

    @property
    def slope(self) -> float:
        """Linear trend over buffer (positive = rising)."""
        if len(self._buf) < 2:
            return 0.0
        x = np.arange(len(self._buf), dtype=float)
        y = np.array(self._buf, dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def last(self) -> float:
        if self._buf:
            return self._buf[-1]
        return 0.0


@dataclass
class GestureEvent:
    channel: str
    state: str        # "onset" | "active" | "offset"
    frame_idx: int
    value: float


class GestureSegmenter:
    """
    Hysteresis-based gesture onset/offset detector for a single channel.

    onset  : signal crosses upper threshold
    offset : signal falls below lower threshold (hysteresis gap prevents flicker)
    """

    def __init__(self, channel: str,
                 upper_thr: float, lower_thr: float,
                 min_duration_frames: int = 3):
        self.channel  = channel
        self.upper    = upper_thr
        self.lower    = lower_thr
        self.min_dur  = min_duration_frames

        self._active = False
        self._dur    = 0
        self._frame  = 0

    def update(self, value: float) -> Optional[GestureEvent]:
        self._frame += 1
        event = None

        if not self._active:
            if value >= self.upper:
                self._active = True
                self._dur    = 1
                event = GestureEvent(self.channel, "onset",
                                     self._frame, value)
        else:
            self._dur += 1
            if value < self.lower:
                if self._dur >= self.min_dur:
                    event = GestureEvent(self.channel, "offset",
                                         self._frame, value)
                self._active = False
                self._dur    = 0
            else:
                event = GestureEvent(self.channel, "active",
                                     self._frame, value)

        return event


class TemporalSmoother:
    """
    Manages per-channel SignalBuffers for all non-manual features.
    Call update() each frame with the latest feature values,
    then read smoothed values from the channel buffers.
    """

    CHANNELS = [
        "left_brow_height", "right_brow_height", "interbrow_distance",
        "mean_ear", "left_ear", "right_ear",
        "gaze_x", "gaze_y",
        "mar", "lip_open", "lip_spread", "lip_protrusion",
        "shoulder_bilateral_raise", "shoulder_lateral_lean", "shoulder_unilateral_shrug",
        "head_pitch", "head_yaw", "head_roll",
        "flow_global_mag",
    ]

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg   = config
        w          = config.thresholds.smoother_window
        alpha      = config.thresholds.smoother_alpha

        self.buffers: Dict[str, SignalBuffer] = {
            ch: SignalBuffer(window=w, alpha=alpha)
            for ch in self.CHANNELS
        }

        self.segmenters: Dict[str, GestureSegmenter] = {
            "both_raised":      GestureSegmenter("brow_raise",   0.55, 0.40, 3),
            "furrowed":         GestureSegmenter("brow_furrow",  0.45, 0.30, 3),
            "mouth_open":       GestureSegmenter("mouth_open",   0.55, 0.35, 4),
            "is_shrugging":     GestureSegmenter("shrug",        0.55, 0.35, 3),
            "is_nodding":       GestureSegmenter("nod",          0.55, 0.35, 2),
            "is_shaking":       GestureSegmenter("shake",        0.55, 0.35, 2),
        }

        self.events: List[GestureEvent] = []

        log.info("TemporalSmoother ready (%d channels, %d segmenters).",
                 len(self.CHANNELS), len(self.segmenters))

    def update(self, values: Dict[str, float],
               discrete: Dict[str, bool]) -> List[GestureEvent]:
        """
        Parameters
        ----------
        values   : dict mapping channel names to scalar float values
        discrete : dict mapping segmenter keys to bool activation values

        Returns list of GestureEvents fired this frame.
        """
        # Update signal buffers
        for ch, buf in self.buffers.items():
            if ch in values:
                buf.update(values[ch])

        # Run segmenters on discrete channels
        fired: List[GestureEvent] = []
        for key, seg in self.segmenters.items():
            val = 1.0 if discrete.get(key, False) else 0.0
            ev  = seg.update(val)
            if ev:
                fired.append(ev)
                self.events.append(ev)

        return fired

    def get_smoothed(self, channel: str,
                     mode: str = "ema") -> float:
        """Return smoothed value for a channel. mode: 'ema' or 'ma'."""
        buf = self.buffers.get(channel)
        if buf is None:
            return 0.0
        return buf.ema if mode == "ema" else buf.moving_average

    def get_slope(self, channel: str) -> float:
        buf = self.buffers.get(channel)
        return buf.slope if buf else 0.0
