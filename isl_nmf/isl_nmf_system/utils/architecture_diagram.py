"""
utils/architecture_diagram.py
================================
Generates a publication-quality architecture diagram of the
ISL NMF pipeline as a PNG image.

Run: python -m utils.architecture_diagram
Produces: architecture_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

def render_architecture(output_path: str = "architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel="", color="#1f6feb", textcolor="white", fontsize=9):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="#58a6ff",
                               linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                color=textcolor, fontsize=fontsize,
                fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    color="#8b949e", fontsize=7, zorder=4)

    def arrow(x1, y1, x2, y2, color="#58a6ff"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=16),
                    zorder=5)

    def label(x, y, text, color="#8b949e", size=7.5):
        ax.text(x, y, text, ha="center", va="center",
                color=color, fontsize=size, style="italic", zorder=4)

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(10, 10.5,
            "ISL Non-Manual Feature Extraction — System Architecture",
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold")

    # ── Row 1: Input ────────────────────────────────────────────────────────
    box(0.3, 8.6, 2.8, 0.9, "Webcam / Video", "OpenCV VideoCapture", "#388bfd")
    arrow(3.1, 9.05, 4.0, 9.05)

    # ── Row 1: Shared inference ─────────────────────────────────────────────
    box(4.0, 8.6, 3.5, 0.9, "FaceLandmarkExtractor",
        "MediaPipe FaceMesh (478 pts) + Pose (33 pts)", "#1f6feb")
    arrow(7.5, 9.05, 8.4, 9.05)
    label(7.95, 9.35, "FaceLandmarkResult")

    # ── Row 1: FusionEngine wrapper ─────────────────────────────────────────
    box(8.4, 8.6, 3.0, 0.9, "FusionEngine", "Orchestrator — builds FeatureVector", "#238636")

    # ── Row 2: Seven extractors ─────────────────────────────────────────────
    extractors = [
        ("HeadPose\nEstimator",    "solvePnP\npitch/yaw/roll",  0.3),
        ("Eyebrow\nTracker",       "brow height\nfurrow/raise",  2.5),
        ("Eye\nTracker",           "EAR + iris\ngaze vector",    4.7),
        ("Lip Contour\nExtractor", "MAR, spread\nprotrusion",    6.9),
        ("Shoulder\nTracker",      "raise/lean\nshrug (calib)",  9.1),
        ("Optical Flow\nTracker",  "Farnebäck\n6 ROIs",         11.3),
        ("Temporal\nSmoother",     "EMA + gesture\nsegmenter",  13.5),
    ]
    for name, sub, x in extractors:
        box(x, 6.8, 2.0, 1.4, name, sub, "#1f3a5f", fontsize=8)
        arrow(9.9, 8.6, x+1.0, 8.2)   # from FusionEngine down
        arrow(x+1.0, 6.8, x+1.0, 6.2) # down to feature vector

    # ── Row 3: Feature Vector ───────────────────────────────────────────────
    box(0.3, 5.3, 19.0, 0.7,
        "Normalised Feature Vector  FV: K → [0,1]   (29 keys)",
        "both_raised · is_shaking · head_nod · mouth_open · wide_eye · lip_spread · shoulder_bilateral_raise · gaze_forward · flow_active ...",
        "#6e40c9", fontsize=9)
    arrow(9.8, 5.3, 9.8, 4.8)
    label(10.4, 5.05, "Dict[str, float]")

    # ── Row 4: Semantic Fusion Graph ────────────────────────────────────────
    box(3.5, 3.8, 12.5, 0.9,
        "⬡  Semantic Fusion Graph  (Novel Contribution)",
        "G=(V,E,W)  |V|=16 concept nodes  |E|=10 implication/suppression edges  ·  decay → evidence → update → propagate  ·  O(|V|+|E|) per frame",
        "#b08800", textcolor="#ffe680", fontsize=9)

    arrow(9.75, 3.8, 9.75, 3.2)

    # ── Row 5: Outputs ──────────────────────────────────────────────────────
    box(0.3,  2.2, 5.5, 0.9, "TextGenerator",
        "temporal dedup + human labels", "#196127")
    box(6.1,  2.2, 4.0, 0.9, "Explainability\nEngine",
        "feature reasoning + why-output", "#6e40c9")
    box(10.4, 2.2, 4.0, 0.9, "Visualizer",
        "OpenCV overlay + SFG panel", "#1f3a5f")
    box(14.7, 2.2, 4.8, 0.9, "EvaluationModule\n+ AblationStudy",
        "SAS · P/R/F1 · latency", "#3d1f8f")

    for x2 in [3.0, 8.1, 12.4, 17.1]:
        arrow(9.75, 3.8, x2, 3.1)

    # ── Row 6: Final output ─────────────────────────────────────────────────
    box(2.5, 0.5, 14.5, 1.1,
        'QUESTION(type=WH)   NEGATION(active)   EMPHASIS(strong)   TOPIC_SHIFT(true)',
        "Structured ISL Linguistic Token Output — grammar-level interpretation",
        "#0a3069", textcolor="#79c0ff", fontsize=10)
    arrow(9.75, 2.2, 9.75, 1.6)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_items = [
        (mpatches.Patch(color="#388bfd"), "Input / Camera"),
        (mpatches.Patch(color="#1f6feb"), "MediaPipe Inference"),
        (mpatches.Patch(color="#1f3a5f"), "Feature Extractor"),
        (mpatches.Patch(color="#b08800"), "Semantic Fusion Graph (Novel)"),
        (mpatches.Patch(color="#196127"), "Output Generator"),
        (mpatches.Patch(color="#6e40c9"), "Explainability"),
    ]
    ax.legend(handles=[h for h,_ in legend_items],
              labels=[l for _,l in legend_items],
              loc="lower right", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", framealpha=0.9)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Architecture diagram saved: {output_path}")


if __name__ == "__main__":
    render_architecture()
