"""
fusion_engine/semantic_confidence_fusion.py
=============================================
Feature 10: Semantic Confidence Fusion Engine

Fuses multiple confidence signals to produce a single,
reliable composite confidence score for each token.

Confidence signals fused:
  1. Graph node weight (SFG activation strength)
  2. Feature evidence strength (how strongly features fired)
  3. Temporal consistency (Kalman stability estimate)
  4. Cross-channel agreement (how many channels agree)
  5. Grammar rule conformance (does it match ISL rules)
  6. Historical frequency (how often this token is seen)

Fusion method: Dempster-Shafer evidence theory (simplified)
Each signal contributes a "belief mass" to the final score.
The fused score is more reliable than any single signal alone.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
from config.config import LinguisticTokens
from semantic_graph.semantic_graph_builder import EVIDENCE_MAP

T = LinguisticTokens

# Belief weights for each evidence source
BELIEF_WEIGHTS = {
    "graph_weight":      0.30,
    "feature_evidence":  0.25,
    "temporal_stable":   0.20,
    "cross_channel":     0.15,
    "grammar_conform":   0.10,
}


@dataclass
class FusedConfidence:
    token: str
    fused_score: float         # 0-1 composite
    fused_pct: int             # 0-100
    grade: str                 # A/B/C/D/F
    components: Dict[str, float]
    explanation: str


class SemanticConfidenceFusionEngine:
    """
    Multi-source confidence fusion using weighted belief aggregation.
    Produces more reliable confidence estimates than single-source scoring.
    """

    HISTORY = 15

    def __init__(self):
        # Per-token activation history for temporal stability
        self._history: Dict[str, deque] = {}

    def _get_history(self, token: str) -> deque:
        if token not in self._history:
            self._history[token] = deque(maxlen=self.HISTORY)
        return self._history[token]

    def fuse(self,
             active_tokens: List[str],
             feature_vector: Dict[str, float],
             graph_weights: Dict[str, float],
             grammar_scores: Dict[str, float] = None) -> Dict[str, FusedConfidence]:
        """
        Compute fused confidence for all active tokens.
        """
        # Update history
        active_set = set(active_tokens)
        for tok, w in graph_weights.items():
            self._get_history(tok).append(w)

        results = {}
        for token in active_tokens:
            if token == T.NEUTRAL:
                continue

            components = {}

            # 1. Graph weight (normalised)
            gw = float(graph_weights.get(token, 0.0))
            components["graph_weight"] = min(gw, 1.0)

            # 2. Feature evidence strength
            evidence_pairs = EVIDENCE_MAP.get(token, [])
            if evidence_pairs:
                total_a = sum(a for _, a in evidence_pairs)
                weighted = sum(
                    a * float(feature_vector.get(k, 0.0))
                    for k, a in evidence_pairs
                )
                components["feature_evidence"] = min(weighted / (total_a + 1e-9), 1.0)
            else:
                components["feature_evidence"] = gw

            # 3. Temporal stability (Kalman-like history smoothness)
            hist = list(self._get_history(token))
            if len(hist) >= 5:
                arr = np.array(hist[-10:])
                stability = float(1.0 - np.std(arr) * 2)
                components["temporal_stable"] = max(0.0, min(stability, 1.0))
            else:
                components["temporal_stable"] = 0.5

            # 4. Cross-channel agreement
            channels_active = sum(
                1 for k, _ in evidence_pairs
                if float(feature_vector.get(k, 0.0)) > 0.30
            )
            total_ch = max(len(evidence_pairs), 1)
            components["cross_channel"] = min(channels_active / total_ch, 1.0)

            # 5. Grammar conformance
            if grammar_scores:
                components["grammar_conform"] = float(grammar_scores.get(token, 0.5))
            else:
                components["grammar_conform"] = 0.5

            # Dempster-Shafer fusion (simplified weighted sum)
            fused = sum(
                BELIEF_WEIGHTS[k] * v
                for k, v in components.items()
                if k in BELIEF_WEIGHTS
            )
            fused = min(max(fused, 0.0), 1.0)
            pct   = int(fused * 100)

            grade = "A" if pct >= 85 else \
                    "B" if pct >= 70 else \
                    "C" if pct >= 55 else \
                    "D" if pct >= 40 else "F"

            # Explanation
            top_component = max(components, key=components.get)
            explanation = (
                f"Fused from {len(components)} sources. "
                f"Strongest: {top_component}={components[top_component]:.2f}"
            )

            results[token] = FusedConfidence(
                token       = token,
                fused_score = fused,
                fused_pct   = pct,
                grade       = grade,
                components  = components,
                explanation = explanation,
            )

        return results
