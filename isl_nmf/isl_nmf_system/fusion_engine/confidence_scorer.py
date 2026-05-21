"""
fusion_engine/confidence_scorer.py
=====================================
Confidence Score Generator for each active semantic token.

Confidence = weighted combination of:
  1. feature_strength    — how strongly the primary features fired (0-1)
  2. temporal_consistency — how stable the token has been over recent frames (0-1)
  3. multi_signal_agreement — how many independent channels agree (0-1)

Output example:
  QUESTION(type=WH)  →  confidence: 87%
  NEGATION(active)   →  confidence: 94%
"""

from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
from semantic_graph.semantic_graph_builder import EVIDENCE_MAP
from config.config import LinguisticTokens

T = LinguisticTokens

# Weights for the three confidence components
W_STRENGTH    = 0.45
W_CONSISTENCY = 0.35
W_AGREEMENT   = 0.20


@dataclass
class TokenConfidence:
    token: str
    confidence: float           # 0.0 - 1.0
    confidence_pct: int         # 0 - 100
    feature_strength: float
    temporal_consistency: float
    multi_signal_agreement: float
    label: str                  # e.g. "HIGH" / "MEDIUM" / "LOW"


class ConfidenceScorer:
    """
    Computes per-token confidence scores using a sliding window
    of historical graph weights for temporal consistency.
    """

    HISTORY_LEN = 20   # frames of history

    def __init__(self):
        # Per-token weight history for consistency calculation
        self._history: Dict[str, deque] = {}

    def _get_history(self, token: str) -> deque:
        if token not in self._history:
            self._history[token] = deque(maxlen=self.HISTORY_LEN)
        return self._history[token]

    def score(self,
              active_tokens: List[str],
              feature_vector: Dict[str, float],
              graph_weights: Dict[str, float]) -> Dict[str, TokenConfidence]:
        """
        Score all currently active tokens.
        Also update history for ALL tokens (including inactive ones).
        """
        # Update history for all tokens
        for token, weight in graph_weights.items():
            self._get_history(token).append(weight)

        results: Dict[str, TokenConfidence] = {}

        for token in active_tokens:
            if token == T.NEUTRAL:
                continue

            # ── 1. Feature strength ──────────────────────────────────
            evidence_pairs = EVIDENCE_MAP.get(token, [])
            if evidence_pairs:
                total_alpha = sum(a for _, a in evidence_pairs)
                weighted_sum = sum(
                    a * float(feature_vector.get(k, 0.0))
                    for k, a in evidence_pairs
                )
                strength = min(weighted_sum / (total_alpha + 1e-9), 1.0)
            else:
                strength = graph_weights.get(token, 0.0)

            # ── 2. Temporal consistency ──────────────────────────────
            hist = list(self._get_history(token))
            if len(hist) >= 3:
                import numpy as np
                consistency = float(np.mean([w > 0.5 for w in hist[-10:]]))
            else:
                consistency = 0.5

            # ── 3. Multi-signal agreement ────────────────────────────
            # Count how many distinct feature channels are above 0.3
            channels_active = sum(
                1 for k, _ in evidence_pairs
                if float(feature_vector.get(k, 0.0)) > 0.30
            )
            total_channels = max(len(evidence_pairs), 1)
            agreement = min(channels_active / total_channels, 1.0)

            # ── Final confidence ─────────────────────────────────────
            confidence = (
                W_STRENGTH    * strength +
                W_CONSISTENCY * consistency +
                W_AGREEMENT   * agreement
            )
            confidence = min(max(confidence, 0.0), 1.0)
            pct = int(confidence * 100)

            label = "HIGH" if pct >= 75 else ("MEDIUM" if pct >= 50 else "LOW")

            results[token] = TokenConfidence(
                token                  = token,
                confidence             = confidence,
                confidence_pct         = pct,
                feature_strength       = strength,
                temporal_consistency   = consistency,
                multi_signal_agreement = agreement,
                label                  = label,
            )

        return results
