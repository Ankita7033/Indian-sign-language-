"""
visualizer/research_dashboard.py
===================================
Feature 7: Research Dashboard Panel

Professional real-time monitoring panel rendered on the video frame.
Shows system health, performance metrics, feature channel status,
and session statistics — exactly like a production AI monitoring dashboard.

Panel sections:
  ┌─────────────────────────────────────────────────────┐
  │  ISL NMF Research Dashboard                         │
  ├──────────────┬──────────────┬───────────────────────┤
  │  PERFORMANCE │  CHANNELS    │  SESSION STATS        │
  │  FPS: 28.4   │  EYE:  ████  │  Frames: 1,240        │
  │  Lat: 18ms   │  BROW: ████  │  Tokens: 47           │
  │  P99: 34ms   │  HEAD: ██░░  │  Captions: 12         │
  │              │  LIP:  ████  │  Adapt: 6/8 ch        │
  └──────────────┴──────────────┴───────────────────────┘
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque


class ResearchDashboard:
    """
    Professional monitoring dashboard for the ISL NMF system.
    Rendered as a semi-transparent panel on the video frame.
    """

    def __init__(self):
        self._fps_buf: deque = deque(maxlen=30)
        self._lat_buf: deque = deque(maxlen=100)
        self._last_t = time.perf_counter()
        self._session_start = time.time()
        self._frame_count = 0
        self._token_count = 0
        self._caption_count = 0
        self._adapt_status = "0/8"

    def update(self, latency_ms: float,
               new_tokens: bool = False,
               new_caption: bool = False,
               adapt_status: str = ""):
        now = time.perf_counter()
        dt  = now - self._last_t
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
        self._last_t = now
        self._lat_buf.append(latency_ms)
        self._frame_count += 1
        if new_tokens:   self._token_count   += 1
        if new_caption:  self._caption_count += 1
        if adapt_status: self._adapt_status   = adapt_status

    def render(self, canvas: np.ndarray,
               feature_vector: Dict[str, float],
               active_tokens: List[str],
               x: int = 5, y: int = 5,
               width: int = 320,
               height: int = 170) -> np.ndarray:

        # Semi-transparent background
        ov = canvas.copy()
        cv2.rectangle(ov, (x, y), (x+width, y+height), (10, 10, 18), -1)
        cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x+width, y+height), (60, 120, 200), 1)

        # Title bar
        cv2.rectangle(canvas, (x, y), (x+width, y+18), (30, 60, 120), -1)
        cv2.putText(canvas, "  ISL NMF Research Dashboard",
                    (x+4, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 220, 255), 1)

        # ── Column 1: Performance ──────────────────────────────────
        cx1 = x + 6
        cy  = y + 28

        fps   = float(np.mean(self._fps_buf)) if self._fps_buf else 0
        lat   = float(np.mean(self._lat_buf)) if self._lat_buf else 0
        p99   = float(np.percentile(list(self._lat_buf), 99)) if len(self._lat_buf) > 5 else 0

        cv2.putText(canvas, "PERFORMANCE",
                    (cx1, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (150, 200, 255), 1)
        cy += 14
        fps_col = (50,220,50) if fps >= 20 else (50,50,220)
        lat_col = (50,220,50) if lat <= 25  else (50,180,220)
        for label, val, unit, col in [
            ("FPS", f"{fps:.1f}", "",   fps_col),
            ("Lat", f"{lat:.1f}", "ms", lat_col),
            ("P99", f"{p99:.1f}", "ms", (180,180,180)),
        ]:
            cv2.putText(canvas, f"{label}: {val}{unit}",
                        (cx1, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        col, 1)
            cy += 12

        # ── Column 2: Feature channels ─────────────────────────────
        cx2 = x + 112
        cy2 = y + 28
        cv2.putText(canvas, "CHANNELS",
                    (cx2, cy2), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (150, 200, 255), 1)
        cy2 += 14

        channels = [
            ("EYE",  max(feature_vector.get("mean_ear",0), feature_vector.get("wide_eye",0))),
            ("BROW", max(feature_vector.get("left_brow_raise",0), feature_vector.get("right_brow_raise",0))),
            ("HEAD", max(feature_vector.get("head_nod",0), feature_vector.get("is_shaking",0))),
            ("LIP",  feature_vector.get("mouth_open",0)),
            ("SHO",  feature_vector.get("shoulder_bilateral_raise",0)),
            ("FLOW", min(feature_vector.get("flow_active",0), 1.0)),
        ]
        bar_max = 80
        for name, val in channels:
            bw = int(min(val, 1.0) * bar_max)
            col = (50,220,50) if val > 0.5 else (50,180,180) if val > 0.2 else (80,80,80)
            cv2.rectangle(canvas, (cx2+28, cy2-8), (cx2+28+bar_max, cy2), (30,30,30), -1)
            cv2.rectangle(canvas, (cx2+28, cy2-8), (cx2+28+bw, cy2), col, -1)
            cv2.putText(canvas, name, (cx2, cy2-1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1)
            cy2 += 12

        # ── Column 3: Session stats ────────────────────────────────
        cx3 = x + 222
        cy3 = y + 28
        elapsed = int(time.time() - self._session_start)
        mm, ss  = divmod(elapsed, 60)

        cv2.putText(canvas, "SESSION",
                    (cx3, cy3), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (150, 200, 255), 1)
        cy3 += 14

        stats = [
            ("Time",  f"{mm:02d}:{ss:02d}"),
            ("Frames",str(self._frame_count)),
            ("Tokens",str(self._token_count)),
            ("Capts", str(self._caption_count)),
            ("Adapt", self._adapt_status),
            ("Mode",  "LIVE"),
        ]
        for label, val in stats:
            cv2.putText(canvas, f"{label}: {val}",
                        (cx3, cy3), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (200,200,200), 1)
            cy3 += 12

        # Active tokens summary
        if active_tokens and active_tokens != ["NEUTRAL"]:
            tok_text = active_tokens[0].split("(")[0][:10]
            cv2.putText(canvas, f"Active: {tok_text}",
                        (cx1, y+height-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (50,220,150), 1)

        return canvas
