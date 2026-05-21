"""
generate_qualitative_examples.py
==================================
Generates Fig. 5. Qualitative semantic activation examples for the research paper.
Stores the output images directly in research_paper_results/figures.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import shutil

def render_qualitative_diagram():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # c:\Users\ASUS\Downloads\isl_nmf_final\isl_nmf\isl_nmf_system
    workspace_root = os.path.dirname(os.path.dirname(base_dir)) # c:\Users\ASUS\Downloads\isl_nmf_final
    
    figures_dirs = [
        os.path.join(workspace_root, "research_paper_results", "figures"),
        os.path.join(os.path.dirname(base_dir), "research_paper_results", "figures")
    ]
    
    for f_dir in figures_dirs:
        os.makedirs(f_dir, exist_ok=True)
        
    figures_dir = figures_dirs[0]
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 10.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    
    # Title
    ax.text(10.5, 10.1, 
            "Fig. 5. Qualitative Semantic Activation & Graph Propagation Examples", 
            ha="center", va="center", color="white", fontsize=15, fontweight="bold")
    
    # Drawing helpers
    def card(x, y, w, h, border_color="#30363d"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor="#161b22", edgecolor=border_color,
                              linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        
    def section_header(x, y, text, color="#58a6ff"):
        ax.text(x, y, text, ha="left", va="center", color=color, 
                fontsize=9.5, fontweight="bold", zorder=3)
        
    def progress_bar(x, y, w, val, label, val_text, bar_color="#58a6ff"):
        ax.text(x, y + 0.18, label, ha="left", va="center", color="#8b949e", fontsize=8.5, zorder=3)
        # Background track
        track = FancyBboxPatch((x, y - 0.15), w, 0.2, boxstyle="round,pad=0.02",
                               facecolor="#21262d", edgecolor="#30363d", linewidth=0.5, zorder=2)
        ax.add_patch(track)
        # Filled bar
        if val > 0:
            fill_w = w * val
            bar = FancyBboxPatch((x, y - 0.15), fill_w, 0.2, boxstyle="round,pad=0.02",
                                  facecolor=bar_color, edgecolor="none", zorder=3)
            ax.add_patch(bar)
        ax.text(x + w + 0.15, y, val_text, ha="left", va="center", color="white", 
                fontsize=8.5, fontweight="bold", zorder=3)
        
    def node_circle(x, y, r, label, color="#388bfd", textcolor="white", fontsize=7.5):
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor="#58a6ff", linewidth=1.0, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", color=textcolor, 
                fontsize=fontsize, fontweight="bold", zorder=5)
        
    def arrow(x1, y1, x2, y2, color="#58a6ff", linestyle="-", label="", label_pos=0.5, textcolor="white"):
        arrow_style = "-|>" if linestyle == "-" else "->"
        # We can draw standard annotations with arrowprops
        if linestyle == "-":
            ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
                        ha="center", va="center", color=textcolor, fontsize=8, fontweight="bold", zorder=6)
        else:
            # dashed arrow
            ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="--", mutation_scale=10),
                        ha="center", va="center", color=textcolor, fontsize=8, fontweight="bold", zorder=6)
            
    def terminal_box(x, y, w, h, subtitle, output_text, accent_color="#2ea44f"):
        # Terminal frame
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                              facecolor="#0d1117", edgecolor="#30363d", linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        # Header bar
        header = FancyBboxPatch((x, y + h - 0.35), w, 0.35, boxstyle="round,pad=0.02",
                                facecolor="#21262d", edgecolor="none", zorder=3)
        ax.add_patch(header)
        # Small window buttons
        for i, btn_c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
            btn = plt.Circle((x + 0.25 + i*0.2, y + h - 0.18), 0.07, color=btn_c, zorder=4)
            ax.add_patch(btn)
            
        ax.text(x + 1.0, y + h - 0.18, "Explainability Engine Output (GRE)", ha="left", va="center",
                color="#8b949e", fontsize=7.5, style="italic", zorder=4)
        
        # Subtitle and output
        ax.text(x + 0.3, y + h - 0.7, subtitle, ha="left", va="center",
                color="#8b949e", fontsize=8, zorder=3)
        ax.text(x + w/2, y + 0.4, f'“{output_text}”', ha="center", va="center",
                color=accent_color, fontsize=12, fontweight="bold", zorder=3)

    # ── CASE A: YN-Question ──────────────────────────────────────────────────
    x_a = 0.5
    card(x_a, 0.5, 6.2, 9.1, border_color="#388bfd")
    # Title box
    rect_a = FancyBboxPatch((x_a + 0.3, 8.4), 5.6, 0.6, boxstyle="round,pad=0.05",
                            facecolor="#1f6feb", edgecolor="#58a6ff", linewidth=1.0, zorder=2)
    ax.add_patch(rect_a)
    ax.text(x_a + 3.1, 8.7, "CONTEXT (a): YN-QUESTION", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)
    
    # Physiological NMF Features
    section_header(x_a + 0.3, 8.0, "1. Physiological NMF Input Signals", color="#58a6ff")
    progress_bar(x_a + 0.3, 7.3, 4.0, 0.82, "Bilateral Brow Raise (hL = hR = 0.31 IPD)", "both_raised = 0.82", bar_color="#58a6ff")
    progress_bar(x_a + 0.3, 6.3, 4.0, 0.43, "Wide Eyes (EAR = 0.38)", "wide_eye = 0.43", bar_color="#58a6ff")
    # forward lean annotation
    rect_lean = FancyBboxPatch((x_a + 0.3, 5.1), 5.6, 0.5, boxstyle="round,pad=0.03",
                               facecolor="#21262d", edgecolor="#30363d", linewidth=0.8, zorder=2)
    ax.add_patch(rect_lean)
    ax.text(x_a + 0.5, 5.35, "Pose Channel: slight forward lean detected (active)", ha="left", va="center",
            color="#8b949e", fontsize=8, style="italic", zorder=3)
    
    # SFG Activation
    section_header(x_a + 0.3, 4.7, "2. Semantic Fusion Graph Propagation", color="#58a6ff")
    node_circle(x_a + 1.2, 3.8, 0.5, "both_raised\n(NMF)", color="#1f3a5f", textcolor="#58a6ff", fontsize=7)
    node_circle(x_a + 1.2, 2.4, 0.5, "wide_eye\n(NMF)", color="#1f3a5f", textcolor="#58a6ff", fontsize=7)
    node_circle(x_a + 4.8, 3.1, 0.65, "QUESTION\n(type=YN)\n[concept]", color="#b08800", textcolor="#ffe680", fontsize=8)
    
    arrow(x_a + 1.8, 3.7, x_a + 4.15, 3.25, color="#66bb6a", label="+0.82", textcolor="#66bb6a")
    arrow(x_a + 1.8, 2.5, x_a + 4.15, 2.95, color="#66bb6a", label="+0.43", textcolor="#66bb6a")
    
    # GRE Speech Output
    section_header(x_a + 0.3, 1.7, "3. Downstream Explainable Output", color="#58a6ff")
    terminal_box(x_a + 0.3, 0.7, 5.6, 1.2, "Identified YN-Question. Context resolved successfully.", "Are you coming?", accent_color="#58a6ff")

    # ── CASE B: Negation ─────────────────────────────────────────────────────
    x_b = 7.4
    card(x_b, 0.5, 6.2, 9.1, border_color="#ef5350")
    # Title box
    rect_b = FancyBboxPatch((x_b + 0.3, 8.4), 5.6, 0.6, boxstyle="round,pad=0.05",
                            facecolor="#ef5350", edgecolor="#ff6b6b", linewidth=1.0, zorder=2)
    ax.add_patch(rect_b)
    ax.text(x_b + 3.1, 8.7, "CONTEXT (b): NEGATION", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)
    
    # Physiological NMF Features
    section_header(x_b + 0.3, 8.0, "1. Physiological NMF Input Signals", color="#ef5350")
    progress_bar(x_b + 0.3, 7.3, 4.0, 0.79, "Sustained Head-Shake (|θyaw| > 18°)", "is_shaking = 0.79", bar_color="#ef5350")
    progress_bar(x_b + 0.3, 6.3, 4.0, 0.81, "Thumbs-Down Polarity (τ- = 0.71)", "thumbs_down = 0.81", bar_color="#ef5350")
    
    # forward lean annotation
    rect_shk = FancyBboxPatch((x_b + 0.3, 5.1), 5.6, 0.5, boxstyle="round,pad=0.03",
                               facecolor="#21262d", edgecolor="#30363d", linewidth=0.8, zorder=2)
    ax.add_patch(rect_shk)
    ax.text(x_b + 0.5, 5.35, "Hand Estimator: active hand-gestures detected (active)", ha="left", va="center",
            color="#8b949e", fontsize=8, style="italic", zorder=3)
    
    # SFG Activation & Suppression
    section_header(x_b + 0.3, 4.7, "2. Semantic Fusion Graph Propagation", color="#ef5350")
    node_circle(x_b + 1.2, 4.0, 0.5, "is_shaking\n(NMF)", color="#1f3a5f", textcolor="#ef5350", fontsize=7)
    node_circle(x_b + 1.2, 2.8, 0.5, "thumbs_down\n(G/E)", color="#1f3a5f", textcolor="#ef5350", fontsize=7)
    
    node_circle(x_b + 4.8, 3.8, 0.65, "NEGATION\n(active)\n[concept]", color="#b08800", textcolor="#ffe680", fontsize=8)
    node_circle(x_b + 4.8, 2.0, 0.55, "AGREEMENT\n[concept]", color="#3d1f8f", textcolor="#a28df2", fontsize=7.5)
    
    arrow(x_b + 1.8, 3.9, x_b + 4.15, 3.85, color="#66bb6a", label="+0.79", textcolor="#66bb6a")
    arrow(x_b + 1.8, 2.9, x_b + 4.15, 3.55, color="#66bb6a", label="+0.81", textcolor="#66bb6a")
    arrow(x_b + 4.8, 3.15, x_b + 4.8, 2.65, color="#ef5350", linestyle="--", label="-0.91", textcolor="#ef5350")
    ax.text(x_b + 5.6, 2.9, "Suppressed", color="#ef5350", fontsize=7, style="italic", fontweight="bold", zorder=5)
    
    # GRE Speech Output
    section_header(x_b + 0.3, 1.7, "3. Downstream Explainable Output", color="#ef5350")
    terminal_box(x_b + 0.3, 0.7, 5.6, 1.2, "Negation resolved. Strong suppression on Agreement.", "No, that is not correct.", accent_color="#ef5350")

    # ── CASE C: Topic Shift ──────────────────────────────────────────────────
    x_c = 14.3
    card(x_c, 0.5, 6.2, 9.1, border_color="#ab47bc")
    # Title box
    rect_c = FancyBboxPatch((x_c + 0.3, 8.4), 5.6, 0.6, boxstyle="round,pad=0.05",
                            facecolor="#ab47bc", edgecolor="#c15ecb", linewidth=1.0, zorder=2)
    ax.add_patch(rect_c)
    ax.text(x_c + 3.1, 8.7, "CONTEXT (c): TOPIC SHIFT", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)
    
    # Physiological NMF Features
    section_header(x_c + 0.3, 8.0, "1. Physiological NMF Input Signals", color="#ab47bc")
    progress_bar(x_c + 0.3, 7.3, 4.0, 0.63, "Head Tilt (ψ > 12°)", "head_tilt = 0.63", bar_color="#ab47bc")
    progress_bar(x_c + 0.3, 6.3, 4.0, 0.55, "Lateral Shoulder Lean (ℓ = 0.18 torso)", "shoulder_lean = 0.55", bar_color="#ab47bc")
    
    # lean annotation
    rect_ln = FancyBboxPatch((x_c + 0.3, 5.1), 5.6, 0.5, boxstyle="round,pad=0.03",
                              facecolor="#21262d", edgecolor="#30363d", linewidth=0.8, zorder=2)
    ax.add_patch(rect_ln)
    ax.text(x_c + 0.5, 5.35, "Torso Estimator: lateral body lean detected (active)", ha="left", va="center",
            color="#8b949e", fontsize=8, style="italic", zorder=3)
    
    # SFG Activation
    section_header(x_c + 0.3, 4.7, "2. Semantic Fusion Graph Propagation", color="#ab47bc")
    node_circle(x_c + 1.2, 3.8, 0.5, "head_tilt\n(NMF)", color="#1f3a5f", textcolor="#ab47bc", fontsize=7)
    node_circle(x_c + 1.2, 2.4, 0.5, "shoulder_lean\n(NMF)", color="#1f3a5f", textcolor="#ab47bc", fontsize=7)
    node_circle(x_c + 4.8, 3.1, 0.65, "TOPIC SHIFT\n(true)\n[concept]", color="#b08800", textcolor="#ffe680", fontsize=8)
    
    arrow(x_c + 1.8, 3.7, x_c + 4.15, 3.25, color="#66bb6a", label="+0.63", textcolor="#66bb6a")
    arrow(x_c + 1.8, 2.5, x_c + 4.15, 2.95, color="#66bb6a", label="+0.55", textcolor="#66bb6a")
    
    # GRE Speech Output
    section_header(x_c + 0.3, 1.7, "3. Downstream Explainable Output", color="#ab47bc")
    terminal_box(x_c + 0.3, 0.7, 5.6, 1.2, "Topic Boundary resolved. Triggering transition.", "Regarding the earlier point...", accent_color="#ab47bc")

    # Legend & Context details
    legend_patch = FancyBboxPatch((1.0, 0.08), 19.0, 0.32, boxstyle="round,pad=0.02",
                                  facecolor="#161b22", edgecolor="#30363d", linewidth=1.0, zorder=2)
    ax.add_patch(legend_patch)
    
    ax.text(1.5, 0.24, "LEGEND: ", color="white", fontsize=8, fontweight="bold", zorder=3)
    ax.text(2.5, 0.24, "⬡ concept node", color="#ffe680", fontsize=8, fontweight="bold", zorder=3)
    ax.text(4.0, 0.24, "⬡ physiological input node", color="#58a6ff", fontsize=8, fontweight="bold", zorder=3)
    ax.text(6.0, 0.24, "──> excitation edge", color="#66bb6a", fontsize=8, fontweight="bold", zorder=3)
    ax.text(8.0, 0.24, "- - > suppression edge", color="#ef5350", fontsize=8, fontweight="bold", zorder=3)
    ax.text(10.5, 0.24, "IPD: Inter-Pupillary Distance", color="#8b949e", fontsize=7.5, style="italic", zorder=3)
    ax.text(13.2, 0.24, "EAR: Eye Aspect Ratio", color="#8b949e", fontsize=7.5, style="italic", zorder=3)
    ax.text(15.5, 0.24, "SFG: Semantic Fusion Graph", color="#b08800", fontsize=7.5, fontweight="bold", zorder=3)
    ax.text(18.2, 0.24, "GRE: Grammar Rules Engine", color="#2ea44f", fontsize=7.5, fontweight="bold", zorder=3)

    plt.tight_layout(pad=0.2)
    
    # Save high-res PNGs to both target dirs
    fig_path = os.path.join(figures_dir, "qualitative_examples.png")
    plt.savefig(fig_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    
    print(f"Fig 5 qualitative semantic examples diagram saved to: {fig_path}")
    
    # Copy to all target dirs
    for target_dir in figures_dirs:
        if target_dir == figures_dir:
            try:
                shutil.copy(fig_path, os.path.join(target_dir, "fig5.png"))
                print(f"Created duplicate copy: fig5.png in {target_dir}")
            except Exception as e:
                print(f"Could not duplicate copy in same dir: {e}")
        else:
            try:
                shutil.copy(fig_path, os.path.join(target_dir, "qualitative_examples.png"))
                shutil.copy(fig_path, os.path.join(target_dir, "fig5.png"))
                print(f"Synced Fig 5 to: {target_dir}")
            except Exception as e:
                print(f"Could not sync Fig 5 to {target_dir}: {e}")

if __name__ == "__main__":
    render_qualitative_diagram()
