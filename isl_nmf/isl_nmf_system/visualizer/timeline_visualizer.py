"""
visualizer/timeline_visualizer.py
====================================
Semantic Timeline Visualization Panel — Feature #5

Shows grammar events across time as a scrolling timeline panel.

Example display:
  ┌─────────────────────────────────────────────┐
  │  TIMELINE  (last 60 frames)                 │
  │  QUESTION_WH   ████████░░░░░░░░░░░░░░░░░░  │
  │  NEGATION      ░░░░░░░░████░░░░░░░░░░░░░░░  │
  │  AGREEMENT     ░░░░░░░░░░░░░░░████░░░░░░░░  │
  │  EMPHASIS      ░░░░░░░░░░░░░░░░░░░████████  │
  └─────────────────────────────────────────────┘

Each row = one token. Filled blocks = active frames.
Rendered as an OpenCV sub-panel on the main video window.
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Deque
from config.config import LinguisticTokens

T = LinguisticTokens

# Tokens to show in timeline (excluding rarely-used ones)
TIMELINE_TOKENS = [
    T.QUESTION_WH, T.QUESTION_YN, T.NEGATION,
    T.EMPHASIS_STRONG, T.AGREEMENT, T.DISAGREEMENT,
    T.DOUBT, T.SURPRISE, T.TOPIC_SHIFT, T.FOCUS,
]

TOKEN_SHORT = {
    T.QUESTION_WH:    "WH-Q  ",
    T.QUESTION_YN:    "YN-Q  ",
    T.NEGATION:       "NEG   ",
    T.EMPHASIS_STRONG:"EMPH  ",
    T.AGREEMENT:      "AGREE ",
    T.DISAGREEMENT:   "DISAGR",
    T.DOUBT:          "DOUBT ",
    T.SURPRISE:       "SURPR ",
    T.TOPIC_SHIFT:    "TOPIC ",
    T.FOCUS:          "FOCUS ",
}

TOKEN_COLORS_BGR = {
    T.QUESTION_WH:    (220, 180, 50),
    T.QUESTION_YN:    (200, 220, 50),
    T.NEGATION:       (50,  50, 220),
    T.EMPHASIS_STRONG:(50, 180, 220),
    T.AGREEMENT:      (50, 220, 100),
    T.DISAGREEMENT:   (80,  80, 200),
    T.DOUBT:          (140, 200, 50),
    T.SURPRISE:       (220, 100, 220),
    T.TOPIC_SHIFT:    (200, 140,  50),
    T.FOCUS:          (50,  220, 220),
}

WINDOW_FRAMES = 90   # frames of history shown


class TimelineVisualizer:
    """
    Maintains per-token activation history and renders a
    scrolling timeline panel onto a BGR canvas.
    """

    def __init__(self, window_frames: int = WINDOW_FRAMES):
        self.window = window_frames
        # Per-token binary activation history
        self._history: Dict[str, Deque[int]] = {
            tok: deque(maxlen=window_frames)
            for tok in TIMELINE_TOKENS
        }

    def update(self, active_tokens: List[str]) -> None:
        """Feed one frame's active tokens."""
        active_set = set(active_tokens)
        for tok in TIMELINE_TOKENS:
            self._history[tok].append(1 if tok in active_set else 0)

    def render(self,
               canvas: np.ndarray,
               x: int, y: int,
               width: int = 400,
               height: int = 240) -> np.ndarray:
        """
        Draw the timeline panel at position (x, y) on canvas.
        Returns the annotated canvas.
        """
        n_tokens = len(TIMELINE_TOKENS)
        row_h    = height // (n_tokens + 1)
        bar_w    = width - 75   # space for label

        # Background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x+width, y+height),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        # Title
        cv2.putText(canvas, "GRAMMAR TIMELINE",
                    (x+8, y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)

        for i, tok in enumerate(TIMELINE_TOKENS):
            ry = y + 20 + i * row_h
            history = list(self._history[tok])
            n_hist  = len(history)

            # Label
            label = TOKEN_SHORT.get(tok, tok[:6])
            col   = TOKEN_COLORS_BGR.get(tok, (180, 180, 180))
            cv2.putText(canvas, label,
                        (x+4, ry+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        col, 1)

            # Background bar
            bx = x + 70
            cv2.rectangle(canvas,
                          (bx, ry+1), (bx+bar_w, ry+row_h-2),
                          (35, 35, 35), -1)

            # History blocks
            if n_hist > 0:
                block_w = max(bar_w // self.window, 2)
                for fi, val in enumerate(history):
                    if val:
                        px = bx + int(fi * bar_w / self.window)
                        cv2.rectangle(canvas,
                                      (px, ry+2),
                                      (px+block_w, ry+row_h-3),
                                      col, -1)

            # Recent activation indicator (rightmost dot)
            if history and history[-1]:
                cv2.circle(canvas,
                           (bx + bar_w - 4, ry + row_h//2),
                           4, col, -1)

        return canvas
