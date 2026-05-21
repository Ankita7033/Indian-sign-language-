"""
temporal_smoother.py
====================
Temporal smoothing utilities for ISL non-manual feature signals.

Provides:
  - KalmanFilter1D: Single-dimensional Kalman filter for scalar feature signals.
  - MovingAverageSmoother: Sliding window moving average.
  - ExponentialMovingSmoother: EMA smoother for fast/responsive signals.
  - FeatureBuffer: Ring buffer for temporal context windows.
  - MultiChannelSmoother: Applies a chosen smoother to a named dict of signals.
"""

from __future__ import annotations

import collections
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────
# 1D KALMAN FILTER
# ─────────────────────────────────────────────
class KalmanFilter1D:
    """
    Minimal scalar Kalman filter for smoothing a single continuous signal.

    State model: x_k = x_{k-1} + w,  w ~ N(0, Q)
    Observation: z_k = x_k + v,       v ~ N(0, R)

    Parameters
    ----------
    process_noise : float
        Q — variance of process noise (how fast the true value can change).
    measurement_noise : float
        R — variance of measurement noise (sensor noise level).
    initial_value : float
        Starting state estimate.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_value: float = 0.0,
    ) -> None:
        self.Q = process_noise
        self.R = measurement_noise
        self.x = initial_value        # state estimate
        self.P = 1.0                  # estimate error covariance
        self.initialized = False

    def update(self, measurement: float) -> float:
        """
        Ingest a new measurement and return the filtered estimate.

        Parameters
        ----------
        measurement : float
            Raw observed value at current timestep.

        Returns
        -------
        float
            Kalman-filtered state estimate.
        """
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update step
        K = P_pred / (P_pred + self.R)      # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1.0 - K) * P_pred

        return self.x

    def reset(self, value: float = 0.0) -> None:
        self.x = value
        self.P = 1.0
        self.initialized = False


# ─────────────────────────────────────────────
# MOVING AVERAGE SMOOTHER
# ─────────────────────────────────────────────
class MovingAverageSmoother:
    """
    Simple sliding-window moving average.

    Parameters
    ----------
    window_size : int
        Number of past frames to average over.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._buffer: collections.deque = collections.deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._buffer.append(value)
        return float(np.mean(self._buffer))

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.window_size


# ─────────────────────────────────────────────
# EXPONENTIAL MOVING AVERAGE SMOOTHER
# ─────────────────────────────────────────────
class ExponentialMovingSmoother:
    """
    Exponential Moving Average (EMA) smoother.

    Output_t = alpha * Measurement_t + (1 - alpha) * Output_{t-1}

    Parameters
    ----------
    alpha : float
        Smoothing factor in [0, 1]. Higher = more responsive to new data.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None

    @property
    def current(self) -> Optional[float]:
        return self._value


# ─────────────────────────────────────────────
# RING BUFFER FOR TEMPORAL CONTEXT
# ─────────────────────────────────────────────
class FeatureBuffer:
    """
    Fixed-length ring buffer that stores recent feature snapshots
    (as dictionaries) for temporal context window access.

    Parameters
    ----------
    max_len : int
        Maximum number of frames to retain.
    """

    def __init__(self, max_len: int = 30) -> None:
        self._buffer: collections.deque = collections.deque(maxlen=max_len)

    def push(self, feature_dict: dict) -> None:
        self._buffer.append(feature_dict)

    def get_window(self, n: Optional[int] = None) -> List[dict]:
        """Return last n frames (or all if n is None)."""
        if n is None:
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def mean_over_key(self, key: str, n: Optional[int] = None) -> float:
        """Compute mean of a scalar feature key over the last n frames."""
        frames = self.get_window(n)
        vals = [f[key] for f in frames if key in f and f[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    def std_over_key(self, key: str, n: Optional[int] = None) -> float:
        """Compute std of a scalar feature key over the last n frames."""
        frames = self.get_window(n)
        vals = [f[key] for f in frames if key in f and f[key] is not None]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_frames: int = 5) -> bool:
        return len(self._buffer) >= min_frames

    def clear(self) -> None:
        self._buffer.clear()


# ─────────────────────────────────────────────
# MULTI-CHANNEL SMOOTHER
# ─────────────────────────────────────────────
class MultiChannelSmoother:
    """
    Applies independent Kalman filters to each named signal channel
    in a feature dictionary.

    Parameters
    ----------
    channels : list[str]
        List of feature key names to smooth.
    mode : str
        One of 'kalman', 'moving_avg', 'ema'.
    process_noise : float
        For Kalman mode.
    measurement_noise : float
        For Kalman mode.
    window_size : int
        For moving average mode.
    alpha : float
        For EMA mode.
    """

    def __init__(
        self,
        channels: List[str],
        mode: str = "kalman",
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        window_size: int = 5,
        alpha: float = 0.3,
    ) -> None:
        self.mode = mode
        self.channels = channels
        self._smoothers: Dict[str, Union[KalmanFilter1D, MovingAverageSmoother, ExponentialMovingSmoother]] = {}

        for ch in channels:
            if mode == "kalman":
                self._smoothers[ch] = KalmanFilter1D(process_noise, measurement_noise)
            elif mode == "moving_avg":
                self._smoothers[ch] = MovingAverageSmoother(window_size)
            elif mode == "ema":
                self._smoothers[ch] = ExponentialMovingSmoother(alpha)
            else:
                raise ValueError(f"Unknown smoothing mode: {mode}")

    def smooth(self, feature_dict: dict) -> dict:
        """
        Apply smoothing to all matching keys in feature_dict.
        Non-matching keys are passed through unchanged.
        Non-numeric values are skipped.

        Parameters
        ----------
        feature_dict : dict
            Raw feature dictionary for the current frame.

        Returns
        -------
        dict
            Feature dictionary with smoothed values.
        """
        smoothed = dict(feature_dict)
        for ch, smoother in self._smoothers.items():
            if ch in feature_dict and isinstance(feature_dict[ch], (int, float, np.floating)):
                raw_val = float(feature_dict[ch])
                smoothed[ch] = smoother.update(raw_val)
        return smoothed

    def reset_all(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()


# ─────────────────────────────────────────────
# BASELINE ESTIMATOR
# ─────────────────────────────────────────────
class BaselineEstimator:
    """
    Estimates the neutral-state baseline of a feature over the first N frames.
    Used for ratio-based thresholding (e.g., eyebrow height, mouth width).

    Parameters
    ----------
    n_frames : int
        Number of frames to accumulate before locking the baseline.
    """

    def __init__(self, n_frames: int = 30) -> None:
        self.n_frames = n_frames
        self._samples: List[float] = []
        self._baseline: Optional[float] = None

    def update(self, value: float) -> Optional[float]:
        """
        Add a sample. Returns the baseline once enough frames are collected,
        otherwise returns None.
        """
        if self._baseline is not None:
            return self._baseline
        self._samples.append(value)
        if len(self._samples) >= self.n_frames:
            self._baseline = float(np.mean(self._samples))
        return self._baseline

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def is_ready(self) -> bool:
        return self._baseline is not None

    def reset(self) -> None:
        self._samples = []
        self._baseline = None


# ─────────────────────────────────────────────
# OSCILLATION DETECTOR (for head shake / nod)
# ─────────────────────────────────────────────
class OscillationDetector:
    """
    Detects oscillatory motion (e.g., head shake = repeated yaw reversals)
    in a scalar signal over a sliding time window.

    Algorithm:
      1. Track sign changes in the first derivative.
      2. Count reversals within window_frames.
      3. If reversals >= min_reversals → oscillation detected.

    Parameters
    ----------
    window_frames : int
        Number of frames to look back.
    min_reversals : int
        Minimum sign reversals to declare oscillation.
    min_amplitude : float
        Minimum absolute change to count as a reversal (filters noise).
    """

    def __init__(
        self,
        window_frames: int = 20,
        min_reversals: int = 2,
        min_amplitude: float = 2.0,
    ) -> None:
        self.window_frames = window_frames
        self.min_reversals = min_reversals
        self.min_amplitude = min_amplitude
        self._history: collections.deque = collections.deque(maxlen=window_frames)
        self._prev_value: Optional[float] = None
        self._prev_direction: int = 0   # +1 or -1

    def update(self, value: float) -> bool:
        """
        Update with new scalar value. Returns True if oscillation is detected.
        """
        self._history.append(value)

        if self._prev_value is None:
            self._prev_value = value
            return False

        delta = value - self._prev_value
        self._prev_value = value

        if abs(delta) < self.min_amplitude:
            return False   # too small to count

        direction = 1 if delta > 0 else -1
        if direction != self._prev_direction and self._prev_direction != 0:
            # A reversal occurred — record it (implicit via counting below)
            pass
        self._prev_direction = direction

        # Count reversals in _history via zero-crossings of diff
        arr = np.array(self._history)
        diffs = np.diff(arr)
        signs = np.sign(diffs)
        # Filter small changes
        signs[np.abs(diffs) < self.min_amplitude] = 0
        # Count sign changes
        reversal_count = int(np.sum(np.abs(np.diff(signs[signs != 0])) > 0))

        return reversal_count >= self.min_reversals

    def reset(self) -> None:
        self._history.clear()
        self._prev_value = None
        self._prev_direction = 0
