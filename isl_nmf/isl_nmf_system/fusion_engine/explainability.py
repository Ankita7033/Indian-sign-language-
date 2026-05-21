"""
fusion_engine/explainability.py
================================
Explainability module — for every active semantic token, produces a
human-readable reasoning string explaining WHICH features caused it
and HOW strongly each one contributed.

Example output:
  NEGATION(active) detected because:
    ✦ head_shake        → 0.92  [primary driver]
    ✦ furrowed_brows    → 0.71  [supporting]
    ◦ lip_spread        → 0.18  [minor]

This satisfies the explainability requirement:
  transparent inference pipeline + feature reasoning logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from config.config import LinguisticTokens
from semantic_graph.semantic_graph_builder import EVIDENCE_MAP

T = LinguisticTokens

# Human-readable names for feature keys
FEATURE_LABELS: Dict[str, str] = {
    "both_raised":              "bilateral brow raise",
    "brow_raise_one":           "unilateral brow raise",
    "furrowed":                 "furrowed brows",
    "brow_velocity":            "rapid brow movement",
    "left_brow_raise":          "left brow raise",
    "right_brow_raise":         "right brow raise",
    "mean_ear":                 "eye openness (EAR)",
    "wide_eye":                 "wide-open eyes",
    "blink":                    "eye blink",
    "gaze_forward":             "forward gaze",
    "gaze_lateral":             "lateral gaze shift",
    "gaze_up":                  "upward gaze",
    "gaze_down":                "downward gaze",
    "head_nod":                 "head nod",
    "is_shaking":               "head shake",
    "head_tilt":                "head tilt",
    "head_pitch_up":            "head pitched up",
    "head_valid":               "head pose stable",
    "mouth_open":               "open mouth",
    "lip_spread":               "spread lips",
    "lip_rounded":              "rounded lips",
    "lip_pursed":               "pursed lips",
    "lip_protrusion":           "lip protrusion",
    "shoulder_bilateral_raise": "bilateral shoulder raise",
    "shoulder_lateral_lean":    "lateral shoulder lean",
    "is_shrugging":             "shoulder shrug",
    "flow_active":              "active face motion",
    "face_stable":              "stable neutral face",
}


@dataclass
class FeatureContribution:
    feature_key: str
    feature_label: str
    raw_signal: float       # value from feature vector [0,1]
    evidence_weight: float  # alpha from EVIDENCE_MAP
    contribution: float     # alpha * signal (unnormalised)
    role: str               # "primary" | "supporting" | "minor"


@dataclass
class TokenExplanation:
    token: str
    node_weight: float
    contributions: List[FeatureContribution]
    reasoning_string: str   # formatted one-liner for display


@dataclass
class ExplainabilityReport:
    frame_idx: int
    token_explanations: List[TokenExplanation]
    summary_lines: List[str]   # ready to print / display

    def to_string(self) -> str:
        return "\n".join(self.summary_lines)


class ExplainabilityEngine:
    """
    Produces per-token explanations from the feature vector
    and graph node weights.
    """

    def explain(self,
                active_tokens: List[str],
                feature_vector: Dict[str, float],
                graph_weights: Dict[str, float],
                frame_idx: int = 0) -> ExplainabilityReport:

        explanations: List[TokenExplanation] = []
        summary: List[str] = []

        summary.append(f"{'─'*52}")
        summary.append(f"  FRAME {frame_idx:05d} — Semantic Explanation")
        summary.append(f"{'─'*52}")

        for token in active_tokens:
            if token == T.NEUTRAL:
                continue

            evidence_pairs = EVIDENCE_MAP.get(token, [])
            total_alpha = sum(a for _, a in evidence_pairs) or 1.0

            contribs: List[FeatureContribution] = []
            for feat_key, alpha in evidence_pairs:
                sig = float(feature_vector.get(feat_key, 0.0))
                contrib = alpha * sig
                label = FEATURE_LABELS.get(feat_key, feat_key)

                # Role classification
                norm_contrib = contrib / total_alpha
                if norm_contrib >= 0.25:
                    role = "primary"
                elif norm_contrib >= 0.10:
                    role = "supporting"
                else:
                    role = "minor"

                contribs.append(FeatureContribution(
                    feature_key    = feat_key,
                    feature_label  = label,
                    raw_signal     = sig,
                    evidence_weight = alpha,
                    contribution   = contrib,
                    role           = role,
                ))

            # Sort by contribution descending
            contribs.sort(key=lambda c: -c.contribution)

            # Build reasoning string (top 3 contributors only)
            top = [c for c in contribs if c.role in ("primary", "supporting")][:3]
            reasoning = " + ".join(c.feature_label for c in top) if top else "accumulated evidence"

            node_w = graph_weights.get(token, 0.0)

            expl = TokenExplanation(
                token           = token,
                node_weight     = node_w,
                contributions   = contribs,
                reasoning_string = reasoning,
            )
            explanations.append(expl)

            # Format summary block
            short_token = token.split("(")[0]
            summary.append(f"\n  ► {token}  [weight={node_w:.3f}]")
            summary.append(f"    Detected because: {reasoning}")
            for c in contribs[:4]:
                icon = "✦" if c.role == "primary" else ("◈" if c.role == "supporting" else "◦")
                bar  = "█" * int(c.raw_signal * 10) + "░" * (10 - int(c.raw_signal * 10))
                summary.append(
                    f"    {icon} {c.feature_label:<28} {bar}  {c.raw_signal:.2f}"
                )

        if not explanations:
            summary.append("  [NEUTRAL — no significant non-manual markers detected]")

        summary.append(f"{'─'*52}")

        return ExplainabilityReport(
            frame_idx          = frame_idx,
            token_explanations = explanations,
            summary_lines      = summary,
        )
