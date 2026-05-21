"""
visualizer/decision_graph_viewer.py
=====================================
Explainable Decision Graph Viewer — Feature #10

Renders a real-time visual graph showing the reasoning chain:

  Eyebrow ↑ ──┐
              ├──► QUESTION(WH) [0.87]
  Gaze fwd ───┘

  Head shake ─┐
              ├──► NEGATION [0.94]
  Furrowed ───┘

Displayed as an OpenCV sub-panel with animated activation bars.
Makes the system an Explainable AI demo — rare in student projects.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from config.config import LinguisticTokens
from semantic_graph.semantic_graph_builder import EVIDENCE_MAP

T = LinguisticTokens

# Short labels for feature keys
FEAT_SHORT = {
    "both_raised":              "Brows ↑↑",
    "brow_raise_one":           "Brow ↑",
    "furrowed":                 "Furrow",
    "is_shaking":               "Shake ←→",
    "head_nod":                 "Nod ↕",
    "head_tilt":                "Tilt",
    "wide_eye":                 "Eyes wide",
    "mouth_open":               "Mouth open",
    "shoulder_bilateral_raise": "Shrug ↑↑",
    "is_shrugging":             "Shrug",
    "gaze_lateral":             "Gaze side",
    "gaze_forward":             "Gaze fwd",
    "lip_spread":               "Lip spread",
    "lip_pursed":               "Lip pursed",
    "face_stable":              "Stable",
}

TOKEN_COL = {
    T.QUESTION_WH:    (220, 200, 50),
    T.QUESTION_YN:    (200, 220, 50),
    T.NEGATION:       (50,  80, 220),
    T.EMPHASIS_STRONG:(50, 180, 220),
    T.AGREEMENT:      (50, 200, 80),
    T.DOUBT:          (180, 50, 180),
    T.SURPRISE:       (50, 200, 200),
}


class DecisionGraphViewer:
    """
    Draws an explainable reasoning graph panel on the video frame.
    Shows which features feed into which active tokens.
    """

    def render(self,
               canvas: np.ndarray,
               active_tokens: List[str],
               feature_vector: Dict[str, float],
               graph_weights: Dict[str, float],
               x: int, y: int,
               width: int = 380,
               height: int = 300) -> np.ndarray:

        if not active_tokens or active_tokens == [T.NEUTRAL]:
            return canvas

        # Background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x+width, y+height),
                      (12, 12, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x+width, y+height),
                      (60, 60, 100), 1)

        cv2.putText(canvas, "DECISION REASONING",
                    (x+8, y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (180, 180, 220), 1)

        # Show top 3 active tokens
        display_tokens = [t for t in active_tokens
                          if t != T.NEUTRAL][:3]

        panel_h_per_token = (height - 25) // max(len(display_tokens), 1)

        for ti, token in enumerate(display_tokens):
            ty = y + 22 + ti * panel_h_per_token
            token_weight = graph_weights.get(token, 0.0)
            tok_col = TOKEN_COL.get(token, (180, 180, 180))

            # Token label + weight bar
            short_tok = token.split("(")[0][:12]
            cv2.putText(canvas, f"► {short_tok}",
                        (x+8, ty+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        tok_col, 1)

            # Weight bar
            bar_w = int(token_weight * 80)
            cv2.rectangle(canvas, (x+130, ty+3),
                          (x+210, ty+13), (40,40,40), -1)
            cv2.rectangle(canvas, (x+130, ty+3),
                          (x+130+bar_w, ty+13), tok_col, -1)
            cv2.putText(canvas, f"{token_weight:.2f}",
                        (x+215, ty+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (200,200,200), 1)

            # Evidence features (top 3)
            evidence = EVIDENCE_MAP.get(token, [])
            evidence_sorted = sorted(
                evidence,
                key=lambda e: e[1] * float(feature_vector.get(e[0], 0.0)),
                reverse=True
            )[:3]

            for ei, (feat_key, alpha) in enumerate(evidence_sorted):
                fy  = ty + 18 + ei * 14
                sig = float(feature_vector.get(feat_key, 0.0))
                feat_label = FEAT_SHORT.get(feat_key, feat_key[:10])

                # Connector line
                cv2.line(canvas,
                         (x+20, ty+12), (x+25, fy+6),
                         (60, 60, 80), 1)

                # Feature label
                feat_col = tok_col if sig > 0.5 else (100, 100, 100)
                cv2.putText(canvas,
                            f"  {feat_label}",
                            (x+25, fy+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            feat_col, 1)

                # Signal bar (mini)
                bw = int(sig * 50)
                cv2.rectangle(canvas,
                              (x+130, fy+1), (x+180, fy+9),
                              (30,30,30), -1)
                cv2.rectangle(canvas,
                              (x+130, fy+1), (x+130+bw, fy+9),
                              feat_col, -1)

                cv2.putText(canvas, f"{sig:.2f}",
                            (x+185, fy+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                            (160,160,160), 1)

        return canvas
