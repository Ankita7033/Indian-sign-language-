"""
semantic_graph/semantic_graph_builder.py
==========================================
Novel Contribution: Semantic Fusion Graph (SFG)

The SFG is a weighted directed graph G = (V, E, W) where:
  V = set of semantic concept nodes (linguistic tokens)
  E = directed edges encoding implication / co-occurrence relationships
  W: V -> [0,1] = real-valued activation weights per node

Per-frame update rule:
  w_v(t) = (1 - decay) * w_v(t-1) + sum_{f in evidence(v)} alpha_f * signal_f(t)

  where:
    decay     = graph_decay_rate  (controls temporal memory)
    alpha_f   = evidence weight for feature channel f -> node v
    signal_f  = normalised activation of feature f at time t

Graph inference:
  A node v is ACTIVE if w_v(t) >= activation_threshold.
  Active nodes are propagated through edges:
    w_u(t) += edge_weight(v,u) * w_v(t)  for all (v,u) in E

This produces a joint linguistic interpretation that considers
inter-feature dependencies rather than independent label outputs.

Mathematical basis:
  The propagation resembles belief propagation on a factor graph,
  restricted to one forward pass per frame for real-time performance.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional

from config.config import (
    SystemConfig, DEFAULT_CONFIG, LinguisticTokens
)
from utils.logger import get_logger

log = get_logger(__name__)
T = LinguisticTokens


# ---------------------------------------------------------------------------
# Evidence mapping: maps feature signals -> graph node names
# ---------------------------------------------------------------------------
# Format: { node_name: [(feature_key, evidence_weight), ...] }
# feature_key must match keys in FeatureVector.to_dict()
EVIDENCE_MAP: Dict[str, List[tuple]] = {
    # DESIGN: Only deliberate, sustained actions fire tokens.
    # brow_velocity and gaze_forward removed — too noisy on neutral faces.
    # Every token needs 2+ independent signals above threshold to accumulate.

    T.QUESTION_WH: [
        ("furrowed",              0.90),   # furrowed brows — deliberate WH question marker
        ("head_pitch_up",         0.45),   # chin lifts or head tilts slightly
        ("head_tilt",             0.35),
    ],
    T.QUESTION_YN: [
        ("both_raised",           0.90),   # raised brows for YN question
        ("wide_eye",              0.45),   # eyes slightly widen
        ("head_tilt",             0.35),   # head tilts forward/sideways
    ],
    T.NEGATION: [
        ("is_shaking",            0.95),   # deliberate head shake required
        ("furrowed",              0.45),
    ],
    T.EMPHASIS_STRONG: [
        ("head_nod",              0.65),
        ("shoulder_bilateral_raise", 0.65),# both needed for strong emphasis
        ("wide_eye",              0.50),
    ],
    T.EMPHASIS_MILD: [
        ("head_nod",              0.55),
        ("brow_raise_one",        0.55),   # unilateral raise + nod
    ],
    T.TOPIC_SHIFT: [
        ("head_tilt",             0.85),   # strong tilt required
        ("shoulder_lateral_lean", 0.55),
        ("gaze_lateral",          0.45),
    ],
    T.DOUBT: [
        ("is_shrugging",          0.90),   # shrug is primary signal
        ("furrowed",              0.55),
    ],
    T.SURPRISE: [
        ("wide_eye",              0.85),   # all three needed simultaneously
        ("both_raised",           0.75),
        ("mouth_open",            0.65),
    ],
    T.AGREEMENT: [
        ("head_nod",              0.98),   # almost purely head nod
    ],
    T.DISAGREEMENT: [
        ("is_shaking",            0.98),   # almost purely head shake
    ],
    T.CONDITIONAL: [
        ("head_tilt",             0.70),   # tilt + unilateral raise
        ("brow_raise_one",        0.65),
        ("gaze_lateral",          0.40),
    ],
    T.TOPIC_MARKER: [
        ("both_raised",           0.65),   # raise + tilt both required
        ("head_tilt",             0.65),
    ],
    T.FOCUS: [
        ("wide_eye",              0.70),   # genuinely wide eyes
        ("both_raised",           0.55),
        ("shoulder_bilateral_raise", 0.45),
    ],
    T.EXCLAMATION: [
        ("wide_eye",              0.70),   # all three required
        ("mouth_open",            0.65),
        ("shoulder_bilateral_raise", 0.60),
    ],
    T.UNCERTAINTY: [
        ("is_shrugging",          0.85),   # shrug is primary
        ("gaze_lateral",          0.50),
        ("furrowed",              0.35),
    ],
    T.NEUTRAL: [
        ("face_stable",           0.80),
    ],
}


# ---------------------------------------------------------------------------
# Edge definitions: implication relationships between semantic nodes
# (source, target, weight)
# ---------------------------------------------------------------------------
GRAPH_EDGES = [
    # WH-question implies focus
    (T.QUESTION_WH,   T.FOCUS,        0.40),
    # Negation strengthens disagreement
    (T.NEGATION,      T.DISAGREEMENT, 0.35),
    # Strong emphasis can imply focus
    (T.EMPHASIS_STRONG, T.FOCUS,      0.30),
    # Surprise activates exclamation
    (T.SURPRISE,      T.EXCLAMATION,  0.45),
    # Doubt implies uncertainty
    (T.DOUBT,         T.UNCERTAINTY,  0.50),
    # Agreement suppresses negation
    (T.AGREEMENT,     T.NEGATION,    -0.50),
    # Disagreement suppresses agreement
    (T.DISAGREEMENT,  T.AGREEMENT,   -0.60),
    # Topic shift can trigger topic marker
    (T.TOPIC_SHIFT,   T.TOPIC_MARKER, 0.40),
    # YN-question can co-occur with agreement
    (T.QUESTION_YN,   T.AGREEMENT,    0.20),
    # Negation can co-occur with emphasis
    (T.NEGATION,      T.EMPHASIS_MILD, 0.25),
]


@dataclass
class SemanticGraphState:
    """Per-frame snapshot of graph activation weights."""
    weights: Dict[str, float] = field(default_factory=dict)
    active_nodes: List[str]   = field(default_factory=list)
    top_token: str            = T.NEUTRAL
    token_sequence: List[str] = field(default_factory=list)


class SemanticFusionGraph:
    """
    Weighted directed graph for multi-channel non-manual feature fusion.

    Usage
    -----
    graph = SemanticFusionGraph(config)
    state = graph.update(feature_vector_dict)
    print(state.token_sequence)
    # -> ["QUESTION(type=WH)", "EMPHASIS(strong)"]
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg   = config
        self.thr   = config.thresholds
        self._decay = self.thr.graph_decay_rate
        self._act_thr = self.thr.graph_activation_threshold

        # Build directed graph
        self._G = nx.DiGraph()
        all_nodes = list(EVIDENCE_MAP.keys())
        self._G.add_nodes_from(all_nodes)

        for src, tgt, w in GRAPH_EDGES:
            self._G.add_edge(src, tgt, weight=w)

        # Node weights (activation state)
        self._weights: Dict[str, float] = {n: 0.0 for n in all_nodes}

        log.info("SemanticFusionGraph built: %d nodes, %d edges.",
                 self._G.number_of_nodes(), self._G.number_of_edges())

    def _compute_evidence(self, node: str,
                           fv: Dict[str, float]) -> float:
        """
        Compute evidence-weighted activation for a single node.

        score = sum_i alpha_i * signal_i
        where alpha_i are evidence weights from EVIDENCE_MAP,
        capped at 1.0.
        """
        evidence_pairs = EVIDENCE_MAP.get(node, [])
        score = 0.0
        total_alpha = 0.0
        for feat_key, alpha in evidence_pairs:
            sig = float(fv.get(feat_key, 0.0))
            score      += alpha * sig
            total_alpha += alpha
        # Normalise by total possible evidence weight
        if total_alpha > 0:
            score /= total_alpha
        return min(score, 1.0)

    def update(self, feature_vector: Dict[str, float]) -> SemanticGraphState:
        """
        One forward pass of graph update.

        Step 1: Decay all node weights
        Step 2: Compute per-node evidence from feature_vector
        Step 3: Add evidence to decayed weights
        Step 4: Propagate through edges (one hop)
        Step 5: Clip weights to [0, 1]
        Step 6: Determine active nodes and token sequence
        """
        # Step 1: Decay
        for node in self._weights:
            self._weights[node] *= (1.0 - self._decay)

        # Step 2+3: Evidence accumulation
        for node in self._weights:
            ev = self._compute_evidence(node, feature_vector)
            self._weights[node] = np.clip(
                self._weights[node] + ev, 0.0, 1.0
            )

        # Step 4: Edge propagation (one hop)
        propagated = dict(self._weights)
        for (src, tgt, data) in self._G.edges(data=True):
            edge_w = data["weight"]
            contribution = edge_w * self._weights[src]
            propagated[tgt] = np.clip(
                propagated[tgt] + contribution, 0.0, 1.0
            )
        self._weights = propagated

        # Step 5: Determine active nodes
        active = [
            node for node, w in self._weights.items()
            if w >= self._act_thr
        ]

        # Exclude NEUTRAL if any other node is active
        if len(active) > 1 and T.NEUTRAL in active:
            active.remove(T.NEUTRAL)

        # Sort by weight descending
        active.sort(key=lambda n: -self._weights[n])

        # Top token
        top = active[0] if active else T.NEUTRAL

        state = SemanticGraphState(
            weights      = dict(self._weights),
            active_nodes = active,
            top_token    = top,
            token_sequence = active if active else [T.NEUTRAL]
        )
        return state

    def reset(self):
        """Reset all node weights to zero."""
        for n in self._weights:
            self._weights[n] = 0.0

    @property
    def graph(self) -> nx.DiGraph:
        return self._G

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)
