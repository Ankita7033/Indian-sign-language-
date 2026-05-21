"""
feature_extractors/eye_gaze_intent.py
========================================
Feature 17: Eye-Gaze Intent Predictor

Predicts communicative intent from eye gaze patterns over time.

In ISL and natural communication, gaze direction encodes:
  - REFERENTIAL: looking at a referent (object/person being discussed)
  - QUESTIONING: sustained forward gaze during questions
  - THINKING:    upward/lateral gaze during processing
  - EMPHASIZING: wide-open forward gaze for emphasis
  - SHIFTING:    lateral gaze indicating topic/referent shift
  - AFFIRMING:   downward nod gaze during agreement

Uses a temporal buffer of gaze vectors to classify intent,
not just instantaneous direction.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
from utils.math_utils import ema_filter

GAZE_HISTORY = 20   # frames


@dataclass
class GazeIntent:
    intent: str          # REFERENTIAL|QUESTIONING|THINKING|EMPHASIZING|SHIFTING|AFFIRMING|NEUTRAL
    confidence: float    # 0-1
    gaze_x: float        # mean horizontal gaze
    gaze_y: float        # mean vertical gaze
    stability: float     # how stable gaze has been (0=erratic, 1=locked)
    description: str


INTENT_DESCRIPTIONS = {
    "REFERENTIAL":  "Looking at a referent (pointing/discussing something)",
    "QUESTIONING":  "Sustained forward gaze (questioning posture)",
    "THINKING":     "Upward/lateral gaze (processing/considering)",
    "EMPHASIZING":  "Wide-open direct gaze (emphasis/importance)",
    "SHIFTING":     "Lateral gaze shift (topic/referent change)",
    "AFFIRMING":    "Downward confirming gaze (agreement/affirmation)",
    "NEUTRAL":      "Normal resting gaze",
}


class EyeGazeIntentPredictor:
    """
    Predicts communicative intent from temporal gaze patterns.
    """

    def __init__(self):
        self._gx_buf: deque = deque(maxlen=GAZE_HISTORY)
        self._gy_buf: deque = deque(maxlen=GAZE_HISTORY)
        self._ear_buf: deque = deque(maxlen=GAZE_HISTORY)
        self._base_gx = 0.0
        self._base_gy = 0.0
        self._base_alpha = 0.01
        self._initialized = False

    def update(self, gaze_x: float, gaze_y: float,
               ear: float) -> GazeIntent:
               
        if not self._initialized:
            self._base_gx = gaze_x
            self._base_gy = gaze_y
            self._initialized = True
            
        self._base_gx = ema_filter(self._base_gx, gaze_x, self._base_alpha)
        self._base_gy = ema_filter(self._base_gy, gaze_y, self._base_alpha)
        
        calib_gx = gaze_x - self._base_gx
        calib_gy = gaze_y - self._base_gy

        self._gx_buf.append(calib_gx)
        self._gy_buf.append(calib_gy)
        self._ear_buf.append(ear)

        if len(self._gx_buf) < 5:
            return GazeIntent("NEUTRAL", 0.5, gaze_x, gaze_y, 0.5,
                               INTENT_DESCRIPTIONS["NEUTRAL"])

        gx_arr = np.array(self._gx_buf)
        gy_arr = np.array(self._gy_buf)
        ear_arr = np.array(self._ear_buf)

        mean_gx  = float(np.mean(gx_arr))
        mean_gy  = float(np.mean(gy_arr))
        std_gx   = float(np.std(gx_arr))
        std_gy   = float(np.std(gy_arr))
        mean_ear = float(np.mean(ear_arr))
        stability = max(0.0, 1.0 - (std_gx + std_gy) * 3)

        # Score each intent
        scores: Dict[str, float] = {}

        # REFERENTIAL: lateral gaze, relatively stable
        scores["REFERENTIAL"] = min(abs(mean_gx) * 2, 1.0) * stability

        # QUESTIONING: forward, stable, normal ear
        fwd = max(0.0, 1.0 - abs(mean_gx)*3 - abs(mean_gy)*2)
        scores["QUESTIONING"] = fwd * stability * 0.8

        # THINKING: upward or lateral with instability
        scores["THINKING"] = (max(0.0, -mean_gy) * 2 +
                               abs(mean_gx) * 1.5) * (1 - stability * 0.5)

        # EMPHASIZING: wide eyes + forward
        scores["EMPHASIZING"] = min(max(0.0, mean_ear - 0.3) * 5, 1.0) * fwd

        # SHIFTING: rapid lateral movement
        scores["SHIFTING"] = min(std_gx * 10, 1.0) if std_gx > 0.1 else 0.0

        # AFFIRMING: slight downward gaze + stable
        scores["AFFIRMING"] = max(0.0, mean_gy * 3) * stability

        scores["NEUTRAL"] = 0.3

        best_intent = max(scores, key=scores.get)
        best_conf   = min(scores[best_intent], 1.0)

        return GazeIntent(
            intent      = best_intent,
            confidence  = best_conf,
            gaze_x      = mean_gx,
            gaze_y      = mean_gy,
            stability   = stability,
            description = INTENT_DESCRIPTIONS[best_intent],
        )
