"""
evaluation/evaluation_metrics_engine.py
=========================================
Feature 1: Evaluation Metrics Engine

Proves system accuracy scientifically with:
  - Frame-level precision, recall, F1 per token class
  - Temporal consistency score (how stable predictions are)
  - Inter-annotator agreement (Cohen's Kappa)
  - Semantic Alignment Score (SAS) — novel metric
  - Confusion matrix per token
  - Latency percentiles: mean, P50, P95, P99

Scientific output: generates a full evaluation report
that can be included directly in a research paper.
"""

import numpy as np
import json, os, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from config.config import LinguisticTokens

T = LinguisticTokens

ALL_TOKENS = [
    T.QUESTION_WH, T.QUESTION_YN, T.NEGATION, T.EMPHASIS_STRONG,
    T.EMPHASIS_MILD, T.TOPIC_SHIFT, T.DOUBT, T.SURPRISE, T.AGREEMENT,
    T.DISAGREEMENT, T.UNCERTAINTY, T.FOCUS, T.EXCLAMATION, T.NEUTRAL,
]


@dataclass
class ClassMetrics:
    token: str
    tp: int = 0; fp: int = 0; fn: int = 0; tn: int = 0

    @property
    def precision(self): return self.tp / (self.tp + self.fp + 1e-9)
    @property
    def recall(self):    return self.tp / (self.tp + self.fn + 1e-9)
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2*p*r / (p + r + 1e-9)
    @property
    def support(self):   return self.tp + self.fn


@dataclass
class EvalReport:
    n_frames: int = 0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    semantic_alignment_score: float = 0.0
    temporal_consistency: float = 0.0
    cohens_kappa: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)

    def to_paper_table(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║        ISL NMF System — Evaluation Report                ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Frames evaluated        : {self.n_frames:<30}║",
            f"║  Macro Precision         : {self.macro_precision:.4f}{'':<25}║",
            f"║  Macro Recall            : {self.macro_recall:.4f}{'':<25}║",
            f"║  Macro F1                : {self.macro_f1:.4f}{'':<25}║",
            f"║  Weighted F1             : {self.weighted_f1:.4f}{'':<25}║",
            f"║  Semantic Alignment (SAS): {self.semantic_alignment_score:.4f}{'':<25}║",
            f"║  Temporal Consistency    : {self.temporal_consistency:.4f}{'':<25}║",
            f"║  Cohen's Kappa           : {self.cohens_kappa:.4f}{'':<25}║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Mean Latency            : {self.mean_latency_ms:.2f} ms{'':<23}║",
            f"║  P50  Latency            : {self.p50_latency_ms:.2f} ms{'':<23}║",
            f"║  P95  Latency            : {self.p95_latency_ms:.2f} ms{'':<23}║",
            f"║  P99  Latency            : {self.p99_latency_ms:.2f} ms{'':<23}║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Per-Class Metrics:                                      ║",
        ]
        for tok, cm in sorted(self.class_metrics.items(), key=lambda x: -x[1].f1):
            if cm.support > 0:
                label = tok.split("(")[0][:18]
                lines.append(
                    f"║  {label:<20} P={cm.precision:.3f} R={cm.recall:.3f} F1={cm.f1:.3f}  ║"
                )
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


class EvaluationMetricsEngine:
    """Scientific evaluation engine for ISL NMF system."""

    def __init__(self):
        self._predictions: Dict[int, List[str]] = {}
        self._ground_truth: Dict[int, Tuple[List[str], float]] = {}
        self._latencies: List[float] = []
        self._token_history: Dict[int, str] = {}  # frame→top token

    def log(self, frame_idx: int, predicted: List[str],
            latency_ms: float, gt_tokens: List[str] = None,
            gt_confidence: float = 1.0):
        self._predictions[frame_idx] = predicted
        self._latencies.append(latency_ms)
        if gt_tokens:
            self._ground_truth[frame_idx] = (gt_tokens, gt_confidence)
        if predicted:
            self._token_history[frame_idx] = predicted[0]

    def compute(self) -> EvalReport:
        r = EvalReport(n_frames=len(self._predictions))

        # Latency
        if self._latencies:
            arr = np.array(self._latencies)
            r.mean_latency_ms = float(np.mean(arr))
            r.p50_latency_ms  = float(np.percentile(arr, 50))
            r.p95_latency_ms  = float(np.percentile(arr, 95))
            r.p99_latency_ms  = float(np.percentile(arr, 99))

        # Temporal consistency: fraction of consecutive frames with same top token
        frames_sorted = sorted(self._token_history.keys())
        if len(frames_sorted) > 1:
            matches = sum(
                1 for i in range(1, len(frames_sorted))
                if self._token_history[frames_sorted[i]] ==
                   self._token_history[frames_sorted[i-1]]
            )
            r.temporal_consistency = matches / (len(frames_sorted) - 1)

        if not self._ground_truth:
            return r

        # Per-class metrics
        cms = {tok: ClassMetrics(token=tok) for tok in ALL_TOKENS}
        sas_scores = []

        for fi, (gt_toks, conf) in self._ground_truth.items():
            pred_set = set(self._predictions.get(fi, [T.NEUTRAL]))
            gt_set   = set(gt_toks)
            # SAS
            union = pred_set | gt_set
            inter = pred_set & gt_set
            sas = (len(inter) / len(union)) * conf if union else conf
            sas_scores.append(sas)
            # TP/FP/FN/TN
            for tok in ALL_TOKENS:
                cm = cms[tok]
                p, g = tok in pred_set, tok in gt_set
                if p and g:   cm.tp += 1
                elif p and not g: cm.fp += 1
                elif not p and g: cm.fn += 1
                else:             cm.tn += 1

        r.class_metrics = cms
        active = [cm for cm in cms.values() if cm.support > 0]
        if active:
            r.macro_precision = float(np.mean([cm.precision for cm in active]))
            r.macro_recall    = float(np.mean([cm.recall    for cm in active]))
            r.macro_f1        = float(np.mean([cm.f1        for cm in active]))
            total = sum(cm.support for cm in active)
            if total:
                r.weighted_f1 = sum(cm.f1 * cm.support for cm in active) / total

        if sas_scores:
            r.semantic_alignment_score = float(np.mean(sas_scores))

        # Cohen's Kappa (binary per-class average)
        kappas = []
        for cm in active:
            n = cm.tp + cm.fp + cm.fn + cm.tn
            if n == 0: continue
            po = (cm.tp + cm.tn) / n
            pe = ((cm.tp+cm.fp)/n)*((cm.tp+cm.fn)/n) + \
                 ((cm.tn+cm.fn)/n)*((cm.tn+cm.fp)/n)
            kappas.append((po - pe) / (1 - pe + 1e-9))
        if kappas:
            r.cohens_kappa = float(np.mean(kappas))

        return r

    def save(self, path: str = "evaluation/results/eval_report.txt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report = self.compute()
        with open(path, "w") as f:
            f.write(report.to_paper_table())
        print(f"Evaluation report saved: {path}")
        return report
