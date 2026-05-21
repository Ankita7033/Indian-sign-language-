"""
evaluation/ablation_study.py
==============================
Ablation study framework for the ISL NMF system.

Measures the contribution of each feature channel to overall
Semantic Alignment Score by selectively zeroing out channels
and comparing against the full-system baseline.

Ablation protocol:
  1. Establish baseline: run full pipeline on recorded frames
  2. For each channel group, zero-out that channel's evidence
     in the feature vector and re-run semantic graph inference
  3. Compare token outputs against ground truth
  4. Contribution = (baseline_SAS - ablated_SAS) / baseline_SAS

Channel groups tested:
  - EYEBROW    : both_raised, furrowed, brow_velocity, brow_raise_one
  - EYE        : mean_ear, wide_eye, gaze_forward, gaze_lateral
  - HEAD_POSE  : head_nod, is_shaking, head_tilt, head_pitch_up
  - LIP        : mouth_open, lip_spread, lip_rounded, lip_pursed
  - SHOULDER   : shoulder_bilateral_raise, is_shrugging
  - OPTICAL_FLOW: flow_active
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from config.config import DEFAULT_CONFIG, SystemConfig
from semantic_graph.semantic_graph_builder import SemanticFusionGraph
from evaluation.evaluation_metrics import (
    EvaluationModule, FrameAnnotation, jaccard_similarity
)
from utils.logger import get_logger

log = get_logger(__name__)


# Channel group definitions
CHANNEL_GROUPS: Dict[str, Set[str]] = {
    "EYEBROW": {
        "both_raised", "brow_raise_one", "furrowed",
        "brow_velocity", "left_brow_raise", "right_brow_raise"
    },
    "EYE": {
        "mean_ear", "wide_eye", "blink",
        "gaze_forward", "gaze_lateral", "gaze_up", "gaze_down"
    },
    "HEAD_POSE": {
        "head_nod", "is_shaking", "head_tilt",
        "head_pitch_up", "head_valid"
    },
    "LIP": {
        "mouth_open", "lip_spread", "lip_rounded",
        "lip_pursed", "lip_protrusion"
    },
    "SHOULDER": {
        "shoulder_bilateral_raise", "shoulder_lateral_lean",
        "is_shrugging"
    },
    "OPTICAL_FLOW": {
        "flow_active", "face_stable"
    },
}


@dataclass
class AblationResult:
    channel_group: str
    baseline_sas: float
    ablated_sas:  float
    relative_drop: float     # (baseline - ablated) / baseline
    rank: int = 0            # 1 = most important


@dataclass
class AblationReport:
    results: List[AblationResult] = field(default_factory=list)
    baseline_sas: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Ablation Study — Channel Contribution to SAS",
            "=" * 55,
            f"  Baseline SAS: {self.baseline_sas:.4f}",
            "",
            f"  {'Channel Group':<18} {'Ablated SAS':>12} {'Drop%':>8} {'Rank':>5}",
            "  " + "-"*47,
        ]
        for r in self.results:
            lines.append(
                f"  {r.channel_group:<18} {r.ablated_sas:>12.4f} "
                f"{r.relative_drop*100:>7.1f}% {r.rank:>5}"
            )
        return "\n".join(lines)


class AblationStudy:
    """
    Runs systematic channel ablation on pre-recorded feature vectors.

    Usage
    -----
    study = AblationStudy(config)
    # Collect (feature_vector, ground_truth_tokens) pairs during a run
    study.add_sample(fv, gt_tokens, confidence)
    # After collection:
    report = study.run()
    print(report.summary())
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._samples: List[Tuple[Dict, List[str], float]] = []
        log.info("AblationStudy ready.")

    def add_sample(self, feature_vector: Dict[str, float],
                   gt_tokens: List[str],
                   confidence: float = 1.0) -> None:
        self._samples.append((dict(feature_vector), gt_tokens, confidence))

    def _run_inference_on_samples(self,
                                  zeroed_channels: Set[str]) -> float:
        """
        Re-run SFG inference with specified channels zeroed.
        Returns mean SAS across all samples.
        """
        sfg = SemanticFusionGraph(self.cfg)
        sas_scores = []
        for fv, gt_tokens, conf in self._samples:
            # Zero out ablated channels
            ablated_fv = dict(fv)
            for ch in zeroed_channels:
                if ch in ablated_fv:
                    ablated_fv[ch] = 0.0

            sfg.reset()
            state = sfg.update(ablated_fv)
            pred  = set(state.token_sequence)
            gt    = set(gt_tokens)
            sas   = jaccard_similarity(pred, gt) * conf
            sas_scores.append(sas)

        return float(np.mean(sas_scores)) if sas_scores else 0.0

    def run(self) -> AblationReport:
        if not self._samples:
            log.warning("No samples collected for ablation study.")
            return AblationReport()

        log.info("Running ablation study on %d samples...", len(self._samples))

        # Baseline
        baseline_sas = self._run_inference_on_samples(set())
        results = []

        for group_name, channels in CHANNEL_GROUPS.items():
            ablated_sas = self._run_inference_on_samples(channels)
            drop = (baseline_sas - ablated_sas) / max(baseline_sas, 1e-9)
            results.append(AblationResult(
                channel_group = group_name,
                baseline_sas  = baseline_sas,
                ablated_sas   = ablated_sas,
                relative_drop = drop
            ))
            log.debug("Ablated %s: SAS=%.4f (drop=%.1f%%)",
                      group_name, ablated_sas, drop * 100)

        # Rank by relative drop (most important = highest drop)
        results.sort(key=lambda r: -r.relative_drop)
        for rank, r in enumerate(results, start=1):
            r.rank = rank

        report = AblationReport(results=results, baseline_sas=baseline_sas)
        log.info("Ablation complete. Top channel: %s", results[0].channel_group)
        return report
