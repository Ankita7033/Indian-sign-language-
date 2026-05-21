"""
semantic_graph/graph_visualizer.py
=====================================
Standalone visualisation of the Semantic Fusion Graph structure
using NetworkX and Matplotlib.

Run: python -m semantic_graph.graph_visualizer
Produces: semantic_graph_structure.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from semantic_graph.semantic_graph_builder import (
    SemanticFusionGraph, GRAPH_EDGES, EVIDENCE_MAP
)
from config.config import LinguisticTokens, DEFAULT_CONFIG

T = LinguisticTokens


# Node color groups
NODE_COLORS = {
    T.QUESTION_WH:     "#4FC3F7",
    T.QUESTION_YN:     "#29B6F6",
    T.NEGATION:        "#EF5350",
    T.ASSERTION:       "#78909C",
    T.EMPHASIS_STRONG: "#FF7043",
    T.EMPHASIS_MILD:   "#FFA726",
    T.TOPIC_SHIFT:     "#AB47BC",
    T.CONDITIONAL:     "#7E57C2",
    T.EXCLAMATION:     "#EC407A",
    T.DOUBT:           "#26A69A",
    T.SURPRISE:        "#FFCA28",
    T.AGREEMENT:       "#66BB6A",
    T.DISAGREEMENT:    "#EF5350",
    T.UNCERTAINTY:     "#26C6DA",
    T.CONFIRMATION:    "#9CCC65",
    T.TOPIC_MARKER:    "#D4E157",
    T.FOCUS:           "#5C6BC0",
    T.BOUNDARY:        "#BDBDBD",
    T.NEUTRAL:         "#757575",
}


def render_graph(output_path: str = "semantic_graph_structure.png"):
    sfg = SemanticFusionGraph(DEFAULT_CONFIG)
    G   = sfg.graph

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # ---- Left: Full graph topology ----
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    ax1.set_title("Semantic Fusion Graph — Topology",
                  color="white", fontsize=13, pad=12)

    pos = nx.spring_layout(G, seed=42, k=2.2)
    node_colors = [NODE_COLORS.get(n, "#888888") for n in G.nodes()]
    node_sizes  = [900 + 200 * G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax1,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)

    # Positive edges (green), negative (red)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax1,
                           edge_color="#66BB6A", arrows=True,
                           arrowsize=18, width=1.5, alpha=0.75,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax1,
                           edge_color="#EF5350", arrows=True,
                           arrowsize=18, width=1.5, alpha=0.75,
                           connectionstyle="arc3,rad=0.1",
                           style="dashed")

    labels = {n: n.split("(")[0].replace("_", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1,
                            font_size=6.5, font_color="white")

    # ---- Right: Evidence heatmap ----
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.set_title("Evidence Channel Weights per Node",
                  color="white", fontsize=13, pad=12)

    all_channels = sorted({ch for pairs in EVIDENCE_MAP.values() for ch, _ in pairs})
    all_nodes    = list(EVIDENCE_MAP.keys())

    matrix = np.zeros((len(all_nodes), len(all_channels)))
    for i, node in enumerate(all_nodes):
        ch_map = dict(EVIDENCE_MAP[node])
        for j, ch in enumerate(all_channels):
            matrix[i, j] = ch_map.get(ch, 0.0)

    im = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
    ax2.set_xticks(range(len(all_channels)))
    ax2.set_xticklabels(all_channels, rotation=65, ha="right",
                        fontsize=6.5, color="white")
    ax2.set_yticks(range(len(all_nodes)))
    node_labels_short = [n.split("(")[0][:15] for n in all_nodes]
    ax2.set_yticklabels(node_labels_short, fontsize=7, color="white")
    ax2.tick_params(colors="white")
    plt.colorbar(im, ax=ax2, label="Evidence Weight",
                 fraction=0.03, pad=0.04).ax.yaxis.label.set_color("white")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Graph structure saved to: {output_path}")


if __name__ == "__main__":
    render_graph()
