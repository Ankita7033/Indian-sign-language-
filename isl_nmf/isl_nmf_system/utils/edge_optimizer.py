"""
utils/edge_optimizer.py
=========================
Edge Deployment Optimizer — Feature #7

Makes the system run efficiently on low-end hardware:
  --edge-mode

Optimizations applied:
  1. Frame skipping   — process every Nth frame
  2. Resolution scaling — downscale before MediaPipe
  3. ROI cropping     — only process face region
  4. Feature caching  — reuse last N frames' slow features
  5. Pipeline timing  — skip optical flow if too slow

Suitable for: tablets, low-end laptops, Raspberry Pi 4+
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EdgeConfig:
    """Configuration for edge deployment optimizations."""
    enabled: bool = False

    # Frame skipping: process 1 out of every N frames
    frame_skip: int = 2          # process every 2nd frame

    # Resolution scaling (0.5 = half resolution)
    scale_factor: float = 0.6

    # Max allowed processing time per frame (ms)
    # If exceeded, skip optical flow next frame
    max_proc_time_ms: float = 50.0

    # Skip optical flow when budget is tight
    skip_flow: bool = False

    # Target FPS for adaptive throttling
    target_fps: int = 20


class EdgeOptimizer:
    """
    Wraps the processing pipeline with edge-mode optimizations.

    Usage:
      optimizer = EdgeOptimizer(EdgeConfig(enabled=True))
      # In frame loop:
      frame = optimizer.preprocess(raw_frame)
      if optimizer.should_process(frame_idx):
          result = engine.process_frame(frame)
          optimizer.record_timing(result.process_time_ms)
      else:
          result = optimizer.last_result  # reuse previous
    """

    def __init__(self, config: EdgeConfig = EdgeConfig()):
        self.cfg = config
        self.last_result = None
        self._timings = []
        self._skip_flow_counter = 0
        log.info("EdgeOptimizer: enabled=%s scale=%.1f skip=%d",
                 config.enabled, config.scale_factor, config.frame_skip)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame for faster processing."""
        if not self.cfg.enabled or self.cfg.scale_factor >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = int(w * self.cfg.scale_factor)
        new_h = int(h * self.cfg.scale_factor)
        return cv2.resize(frame, (new_w, new_h),
                          interpolation=cv2.INTER_LINEAR)

    def should_process(self, frame_idx: int) -> bool:
        """Return True if this frame should be fully processed."""
        if not self.cfg.enabled:
            return True
        return frame_idx % self.cfg.frame_skip == 0

    def record_timing(self, proc_time_ms: float) -> None:
        """Record processing time for adaptive throttling."""
        self._timings.append(proc_time_ms)
        if len(self._timings) > 30:
            self._timings.pop(0)

        # Adaptive: if we're over budget, skip optical flow
        if len(self._timings) >= 5:
            avg = sum(self._timings[-5:]) / 5
            self.cfg.skip_flow = avg > self.cfg.max_proc_time_ms

    def get_stats(self) -> str:
        """Return current performance stats string."""
        if not self._timings:
            return "EdgeOptimizer: no data"
        avg = sum(self._timings) / len(self._timings)
        fps = 1000.0 / max(avg, 1)
        return (f"Edge: avg={avg:.1f}ms  fps≈{fps:.0f}  "
                f"skip_flow={self.cfg.skip_flow}")
