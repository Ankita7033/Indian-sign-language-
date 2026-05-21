"""
evaluation/evaluation_metrics.py
===================================
Evaluation module for the ISL NMF system.

Computes:
  1. Frame-level token classification metrics
     (precision, recall, F1 per token class)
  2. Latency statistics (mean, median, P95, P99)
  3. Semantic Alignment Score (SAS):
     A novel metric measuring how well the predicted token sequence
     aligns with an annotated reference sequence.
     SAS uses token-level Jaccard similarity averaged across frames,
     weighted by annotation confidence.
  4. Ablation support: per-channel contribution tracking

Ground-truth format
-------------------
The system does NOT use a pre-existing deep-learning dataset.
Instead it uses a protocol-based annotation approach compatible
with ISLRTC annotation guidelines:

  ground_truth = [
    {"frame_idx": int, "tokens": List[str], "confidence": float},
    ...
  ]

where tokens come from the LinguisticTokens vocabulary and
confidence (0-1) reflects annotator agreement.

If no ground truth is provided, the module operates in
LATENCY-ONLY mode.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import defaultdict
import time
import os

from config.config import LinguisticTokens, DEFAULT_CONFIG, SystemConfig
from utils.logger import get_logger

log = get_logger(__name__)
T = LinguisticTokens

ALL_TOKENS = [
    T.QUESTION_WH, T.QUESTION_YN, T.NEGATION, T.ASSERTION,
    T.EMPHASIS_STRONG, T.EMPHASIS_MILD, T.TOPIC_SHIFT,
    T.CONDITIONAL, T.EXCLAMATION, T.DOUBT, T.SURPRISE,
    T.AGREEMENT, T.DISAGREEMENT, T.UNCERTAINTY, T.CONFIRMATION,
    T.TOPIC_MARKER, T.FOCUS, T.BOUNDARY, T.NEUTRAL,
]


@dataclass
class FrameAnnotation:
    frame_idx: int
    tokens: List[str]
    confidence: float = 1.0


@dataclass
class ClassMetrics:
    token: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-9)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-9)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + 1e-9)


@dataclass
class EvaluationReport:
    # Per-class metrics
    class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)

    # Macro averages
    macro_precision: float = 0.0
    macro_recall:    float = 0.0
    macro_f1:        float = 0.0

    # Weighted averages (by class frequency)
    weighted_f1: float = 0.0

    # Latency (ms)
    mean_latency:   float = 0.0
    median_latency: float = 0.0
    p95_latency:    float = 0.0
    p99_latency:    float = 0.0
    max_latency:    float = 0.0

    # Semantic Alignment Score
    semantic_alignment_score: float = 0.0

    # Frame counts
    n_frames_evaluated: int = 0
    n_frames_with_gt:   int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ISL NMF System — Evaluation Report",
            "=" * 60,
            f"  Frames evaluated    : {self.n_frames_evaluated}",
            f"  Frames with GT      : {self.n_frames_with_gt}",
            "",
            "  [Classification Metrics]",
            f"  Macro Precision     : {self.macro_precision:.4f}",
            f"  Macro Recall        : {self.macro_recall:.4f}",
            f"  Macro F1            : {self.macro_f1:.4f}",
            f"  Weighted F1         : {self.weighted_f1:.4f}",
            "",
            "  [Latency (ms)]",
            f"  Mean                : {self.mean_latency:.2f}",
            f"  Median              : {self.median_latency:.2f}",
            f"  P95                 : {self.p95_latency:.2f}",
            f"  P99                 : {self.p99_latency:.2f}",
            f"  Max                 : {self.max_latency:.2f}",
            "",
            f"  [Semantic Alignment Score (SAS): {self.semantic_alignment_score:.4f}]",
            "=" * 60,
        ]
        if self.class_metrics:
            lines.append("  Per-class F1:")
            for tok, cm in sorted(self.class_metrics.items(),
                                   key=lambda x: -x[1].f1):
                label = tok.split("(")[0][:20]
                lines.append(f"    {label:<22} P={cm.precision:.3f} "
                             f"R={cm.recall:.3f} F1={cm.f1:.3f}")
        return "\n".join(lines)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Token-level Jaccard similarity for two token sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union)


class EvaluationModule:
    """
    Collects per-frame predictions and computes the full evaluation report.

    Usage
    -----
    evaluator = EvaluationModule(config)
    # During pipeline loop:
    evaluator.log_prediction(frame_idx, predicted_tokens, latency_ms)
    # Optional: load ground truth
    evaluator.load_ground_truth(annotations)
    # At end:
    report = evaluator.compute_report()
    print(report.summary())
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._predictions: Dict[int, List[str]] = {}   # frame_idx -> tokens
        self._latencies:   List[float] = []
        self._ground_truth: Dict[int, FrameAnnotation] = {}
        self._start_time = time.time()

        log.info("EvaluationModule ready.")

    def log_prediction(self, frame_idx: int,
                       predicted_tokens: List[str],
                       latency_ms: float) -> None:
        self._predictions[frame_idx] = predicted_tokens
        self._latencies.append(latency_ms)

    def load_ground_truth(self,
                          annotations: List[FrameAnnotation]) -> None:
        for ann in annotations:
            self._ground_truth[ann.frame_idx] = ann
        log.info("Ground truth loaded: %d annotated frames.", len(annotations))

    def compute_report(self) -> EvaluationReport:
        report = EvaluationReport()
        report.n_frames_evaluated = len(self._predictions)
        report.n_frames_with_gt   = len(self._ground_truth)

        # ---- Latency stats ----
        if self._latencies:
            arr = np.array(self._latencies)
            report.mean_latency   = float(np.mean(arr))
            report.median_latency = float(np.median(arr))
            report.p95_latency    = float(np.percentile(arr, 95))
            report.p99_latency    = float(np.percentile(arr, 99))
            report.max_latency    = float(np.max(arr))

        if not self._ground_truth:
            log.warning("No ground truth loaded — latency-only report.")
            return report

        # ---- Classification metrics ----
        class_metrics: Dict[str, ClassMetrics] = {
            tok: ClassMetrics(token=tok) for tok in ALL_TOKENS
        }

        sas_scores: List[float] = []

        for fi, gt_ann in self._ground_truth.items():
            pred_tokens = set(self._predictions.get(fi, [T.NEUTRAL]))
            gt_tokens   = set(gt_ann.tokens)
            conf        = gt_ann.confidence

            # SAS for this frame
            sas = jaccard_similarity(pred_tokens, gt_tokens) * conf
            sas_scores.append(sas)

            # Per-class TP/FP/FN
            for tok in ALL_TOKENS:
                cm = class_metrics[tok]
                pred_has = tok in pred_tokens
                gt_has   = tok in gt_tokens
                if pred_has and gt_has:
                    cm.tp += 1
                elif pred_has and not gt_has:
                    cm.fp += 1
                elif not pred_has and gt_has:
                    cm.fn += 1

        report.class_metrics = class_metrics

        # Macro averages (over classes that appear in GT)
        active = [cm for cm in class_metrics.values()
                  if cm.tp + cm.fn > 0]
        if active:
            report.macro_precision = float(np.mean([cm.precision for cm in active]))
            report.macro_recall    = float(np.mean([cm.recall    for cm in active]))
            report.macro_f1        = float(np.mean([cm.f1        for cm in active]))

            total_support = sum(cm.tp + cm.fn for cm in active)
            if total_support > 0:
                report.weighted_f1 = float(
                    sum(cm.f1 * (cm.tp + cm.fn) for cm in active) / total_support
                )

        if sas_scores:
            report.semantic_alignment_score = float(np.mean(sas_scores))

        log.info("Evaluation report computed. Macro-F1=%.4f SAS=%.4f",
                 report.macro_f1, report.semantic_alignment_score)
        return report

    def save_report(self, report: EvaluationReport,
                    path: str = "evaluation/results/report.txt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.summary())
        log.info("Report saved to %s", path)
