"""
fusion_engine/temporal_memory.py
==================================
Temporal Grammar Memory Module

Instead of reacting per-frame, this module maintains a sliding
window buffer and only confirms a token when it has been
consistently active for a minimum number of frames.

This produces linguistically stable output:
  frame 1: eyebrow raise (candidate)
  frame 2: eyebrow raise (candidate)
  frame 5: eyebrow raise (CONFIRMED → QUESTION detected)

Uses a Kalman-inspired smoothing approach on token activation
streams to separate genuine gestures from transient noise.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import numpy as np

from config.config import LinguisticTokens

T = LinguisticTokens

# Minimum consecutive frames a token must be active to confirm
MIN_CONFIRMATION_FRAMES = 5
# Frames token must be absent to be cleared
CLEAR_FRAMES = 8
# Kalman process noise
Q = 0.01
# Kalman measurement noise
R = 0.1


@dataclass
class TokenState:
    token: str
    consecutive_active: int = 0
    consecutive_inactive: int = 0
    confirmed: bool = False
    kalman_estimate: float = 0.0
    kalman_error: float = 1.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))


class TemporalMemory:
    """
    Temporal grammar memory with Kalman smoothing per token.

    Call update() every frame with the raw active token list.
    Read confirmed_tokens for stable linguistic output.
    """

    def __init__(self,
                 min_frames: int = MIN_CONFIRMATION_FRAMES,
                 clear_frames: int = CLEAR_FRAMES):
        self.min_frames   = min_frames
        self.clear_frames = clear_frames
        self._states: Dict[str, TokenState] = {}

    def _get_state(self, token: str) -> TokenState:
        if token not in self._states:
            self._states[token] = TokenState(token=token)
        return self._states[token]

    def _kalman_update(self, state: TokenState, measurement: float) -> float:
        """Single-step Kalman filter update for a scalar signal."""
        # Predict
        x_pred = state.kalman_estimate
        p_pred = state.kalman_error + Q
        # Update
        K = p_pred / (p_pred + R)
        x_new = x_pred + K * (measurement - x_pred)
        p_new = (1 - K) * p_pred
        state.kalman_estimate = x_new
        state.kalman_error    = p_new
        return x_new

    def update(self, raw_tokens: List[str],
               graph_weights: Dict[str, float]) -> List[str]:
        """
        Update temporal memory with this frame's raw token list.

        Returns the CONFIRMED stable token list — only tokens that
        have been consistently active for min_frames are included.
        """
        active_set = set(raw_tokens)

        # Update all known tokens
        all_tokens = set(self._states.keys()) | active_set
        for token in all_tokens:
            state = self._get_state(token)
            weight = float(graph_weights.get(token, 0.0))
            smoothed = self._kalman_update(state, weight)
            state.history.append(smoothed)

            if token in active_set:
                state.consecutive_active   += 1
                state.consecutive_inactive  = 0
                if state.consecutive_active >= self.min_frames:
                    state.confirmed = True
            else:
                state.consecutive_inactive += 1
                state.consecutive_active    = 0
                if state.consecutive_inactive >= self.clear_frames:
                    state.confirmed = False

        # Return confirmed tokens sorted by Kalman estimate descending
        confirmed = [
            t for t, s in self._states.items()
            if s.confirmed and t != T.NEUTRAL
        ]
        confirmed.sort(key=lambda t: -self._states[t].kalman_estimate)

        if not confirmed:
            return [T.NEUTRAL]
        return confirmed

    def get_stability(self, token: str) -> float:
        """Returns 0-1 stability score for a token."""
        state = self._states.get(token)
        if not state or not state.history:
            return 0.0
        return float(np.mean(list(state.history)[-10:]))

    def reset(self):
        self._states.clear()
