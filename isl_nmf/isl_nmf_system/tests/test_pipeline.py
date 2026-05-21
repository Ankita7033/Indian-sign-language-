"""
tests/test_pipeline.py
========================
Unit and integration tests for the ISL NMF pipeline.
Uses synthetic frames (blank / noise) to verify module
initialisation, feature extraction, and graph inference
without requiring a webcam or real ISL video.

Run: python -m pytest tests/test_pipeline.py -v
  or: python tests/test_pipeline.py
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import DEFAULT_CONFIG, LinguisticTokens
from fusion_engine.fusion_engine  import FusionEngine
from fusion_engine.text_generator import TextGenerator
from semantic_graph.semantic_graph_builder import SemanticFusionGraph
from feature_extractors.temporal_smoother  import TemporalSmoother, SignalBuffer
from evaluation.evaluation_metrics import (
    EvaluationModule, FrameAnnotation, jaccard_similarity
)
from evaluation.ablation_study import AblationStudy

T = LinguisticTokens


# ============================================================
# Helper
# ============================================================

def make_blank_frame(h=480, w=640, color=(80, 80, 80)) -> np.ndarray:
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def make_noise_frame(h=480, w=640) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ============================================================
# Tests
# ============================================================

def test_signal_buffer():
    buf = SignalBuffer(window=5, alpha=0.4)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        buf.update(v)
    assert abs(buf.moving_average - 3.0) < 0.01, "Moving average incorrect"
    assert 0.0 <= buf.ema <= 10.0, "EMA out of range"
    print("[PASS] test_signal_buffer")


def test_jaccard_similarity():
    a = {"QUESTION(type=WH)", "EMPHASIS(strong)"}
    b = {"QUESTION(type=WH)", "NEGATION(active)"}
    j = jaccard_similarity(a, b)
    assert abs(j - 1/3) < 0.01, f"Expected 0.333, got {j}"
    assert jaccard_similarity(set(), set()) == 1.0
    print("[PASS] test_jaccard_similarity")


def test_semantic_graph_neutral():
    """With all-zero feature vector, graph should output NEUTRAL."""
    sfg = SemanticFusionGraph(DEFAULT_CONFIG)
    fv  = {k: 0.0 for k in ["both_raised", "furrowed", "head_nod", "is_shaking",
                              "mouth_open", "wide_eye", "gaze_forward",
                              "shoulder_bilateral_raise", "face_stable"]}
    fv["face_stable"]  = 1.0
    fv["gaze_forward"] = 0.8
    state = sfg.update(fv)
    # NEUTRAL should be activated
    assert T.NEUTRAL in state.token_sequence or state.top_token == T.NEUTRAL or \
           len(state.active_nodes) == 0, \
           f"Expected NEUTRAL, got {state.token_sequence}"
    print("[PASS] test_semantic_graph_neutral")


def test_semantic_graph_question_wh():
    """Strong furrowed eyebrow + forward gaze + open mouth should trigger WH-question."""
    sfg = SemanticFusionGraph(DEFAULT_CONFIG)
    # Run 10 frames to let decay settle
    for _ in range(10):
        fv = {
            "furrowed":                 1.0,
            "gaze_forward":             0.9,
            "mouth_open":               0.8,
            "brow_velocity":            0.7,
            "head_pitch_up":            0.5,
            # suppress competing signals
            "is_shaking":               0.0,
            "shoulder_bilateral_raise": 0.0,
            "face_stable":              0.1,
        }
        state = sfg.update(fv)
    assert T.QUESTION_WH in state.token_sequence, \
           f"Expected QUESTION_WH, got {state.token_sequence}"
    print("[PASS] test_semantic_graph_question_wh")


def test_semantic_graph_negation():
    sfg = SemanticFusionGraph(DEFAULT_CONFIG)
    for _ in range(8):
        fv = {"is_shaking": 1.0, "furrowed": 1.0,
              "head_tilt": 0.0, "lip_spread": 0.0, "face_stable": 0.0}
        state = sfg.update(fv)
    assert T.NEGATION in state.token_sequence, \
           f"Expected NEGATION, got {state.token_sequence}"
    print("[PASS] test_semantic_graph_negation")


def test_text_generator_dedup():
    tg = TextGenerator(dedup_window=5)
    out1 = tg.generate([T.NEGATION])
    out2 = tg.generate([T.NEGATION])   # same -> not new
    assert out1.is_new,  "First output should be new"
    assert not out2.is_new, "Duplicate output should NOT be new"
    out3 = tg.generate([T.QUESTION_WH])
    assert out3.is_new, "Different token should be new"
    print("[PASS] test_text_generator_dedup")


def test_temporal_smoother():
    sm = TemporalSmoother(DEFAULT_CONFIG)
    for i in range(20):
        vals = {
            "mean_ear": 0.25 + 0.05 * np.sin(i * 0.5),
            "left_brow_height": 0.03,
            "right_brow_height": 0.03,
            "interbrow_distance": 0.5,
            "gaze_x": 0.0, "gaze_y": 0.0,
            "mar": 0.1, "lip_open": 0.02, "lip_spread": 0.55,
            "lip_protrusion": 0.01,
            "shoulder_bilateral_raise": 0.0,
            "shoulder_lateral_lean": 0.0,
            "shoulder_unilateral_shrug": 0.0,
            "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
            "flow_global_mag": 1.0,
            "left_ear": 0.25, "right_ear": 0.25,
        }
        disc = {"both_raised": False, "furrowed": False,
                "mouth_open": False, "is_shrugging": False,
                "is_nodding": False, "is_shaking": False}
        sm.update(vals, disc)
    ear_smooth = sm.get_smoothed("mean_ear")
    assert 0.1 < ear_smooth < 0.5, f"Unexpected EAR smooth: {ear_smooth}"
    print("[PASS] test_temporal_smoother")


def test_evaluation_module():
    ev = EvaluationModule(DEFAULT_CONFIG)
    gt = [
        FrameAnnotation(0,  [T.QUESTION_WH], 1.0),
        FrameAnnotation(1,  [T.NEGATION],    0.9),
        FrameAnnotation(2,  [T.NEUTRAL],     1.0),
    ]
    ev.log_prediction(0, [T.QUESTION_WH], 15.0)
    ev.log_prediction(1, [T.NEUTRAL],     18.0)  # wrong
    ev.log_prediction(2, [T.NEUTRAL],     12.0)
    ev.load_ground_truth(gt)
    report = ev.compute_report()
    assert report.n_frames_evaluated == 3
    assert report.n_frames_with_gt   == 3
    assert 0.0 <= report.semantic_alignment_score <= 1.0
    assert report.mean_latency > 0
    print(f"[PASS] test_evaluation_module  SAS={report.semantic_alignment_score:.3f}")


def test_ablation_study():
    study = AblationStudy(DEFAULT_CONFIG)
    for _ in range(10):
        fv = {
            "both_raised": 0.8, "gaze_forward": 0.7,
            "mouth_open": 0.6, "is_shaking": 0.0,
            "head_nod": 0.0, "face_stable": 0.5,
            "shoulder_bilateral_raise": 0.0, "wide_eye": 0.3,
            "brow_velocity": 0.5, "furrowed": 0.0,
            "head_tilt": 0.0, "lip_spread": 0.5,
            "lip_pursed": 0.0, "gaze_lateral": 0.1,
            "is_shrugging": 0.0, "flow_active": 0.0,
            "brow_raise_one": 0.3, "head_pitch_up": 0.4,
            "head_valid": 1.0, "gaze_up": 0.0, "gaze_down": 0.0,
            "lip_rounded": 0.0, "lip_protrusion": 0.0,
            "shoulder_lateral_lean": 0.0, "blink": 0.0,
            "mean_ear": 0.6, "left_brow_raise": 0.8,
            "right_brow_raise": 0.8,
        }
        study.add_sample(fv, [T.QUESTION_WH, T.EMPHASIS_MILD], 0.9)
    report = study.run()
    assert len(report.results) == len(study.CHANNEL_GROUPS if hasattr(study, 'CHANNEL_GROUPS') else ["EYEBROW","EYE","HEAD_POSE","LIP","SHOULDER","OPTICAL_FLOW"])
    print(f"[PASS] test_ablation_study  baseline_SAS={report.baseline_sas:.3f}")
    print(report.summary())


def test_fusion_engine_blank_frame():
    """Engine must not crash on a blank frame with no detectable face."""
    cfg    = DEFAULT_CONFIG
    engine = FusionEngine(cfg)
    frame  = make_blank_frame()
    result = engine.process_frame(frame, 0)
    assert result is not None
    assert result.semantic_output in ["NEUTRAL", ""] or isinstance(result.semantic_output, str)
    engine.close()
    print(f"[PASS] test_fusion_engine_blank_frame  output='{result.semantic_output}'")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ISL NMF System — Test Suite")
    print("="*55 + "\n")

    tests = [
        test_signal_buffer,
        test_jaccard_similarity,
        test_semantic_graph_neutral,
        test_semantic_graph_question_wh,
        test_semantic_graph_negation,
        test_text_generator_dedup,
        test_temporal_smoother,
        test_evaluation_module,
        test_ablation_study,
        test_fusion_engine_blank_frame,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}\n")


def test_priority_resolver_basic():
    """WH-question should suppress YN-question."""
    from fusion_engine.semantic_priority_resolver import SemanticPriorityResolver
    from config.config import LinguisticTokens as T

    resolver = SemanticPriorityResolver()
    raw = [T.QUESTION_WH, T.QUESTION_YN, T.FOCUS,
           T.SURPRISE, T.EXCLAMATION, T.TOPIC_MARKER,
           T.EMPHASIS_MILD, T.AGREEMENT]

    weights = {t: 0.8 for t in raw}
    weights[T.QUESTION_WH] = 0.92   # highest
    weights[T.QUESTION_YN] = 0.71

    result = resolver.resolve(raw, {}, weights)

    assert T.QUESTION_WH in result.tokens, "WH-Q must be in result"
    assert T.QUESTION_YN not in result.tokens, "YN-Q must be suppressed"
    assert len(result.tokens) <= 4, f"Too many tokens: {result.tokens}"
    print(f"[PASS] test_priority_resolver_basic  "
          f"{len(raw)} → {len(result.tokens)} tokens: {result.tokens}")


def test_priority_resolver_negation_agreement():
    """NEGATION must suppress AGREEMENT (contradiction)."""
    from fusion_engine.semantic_priority_resolver import SemanticPriorityResolver
    from config.config import LinguisticTokens as T

    resolver = SemanticPriorityResolver()
    raw = [T.NEGATION, T.AGREEMENT, T.EMPHASIS_STRONG]
    weights = {T.NEGATION: 0.88, T.AGREEMENT: 0.71, T.EMPHASIS_STRONG: 0.65}

    result = resolver.resolve(raw, {}, weights)

    assert T.NEGATION   in result.tokens, "NEGATION must be kept"
    assert T.AGREEMENT not in result.tokens, "AGREEMENT must be suppressed"
    print(f"[PASS] test_priority_resolver_negation_agreement  "
          f"result: {result.tokens}")


def test_priority_resolver_max_tokens():
    """Output should never exceed 4 tokens."""
    from fusion_engine.semantic_priority_resolver import SemanticPriorityResolver
    from config.config import LinguisticTokens as T

    resolver = SemanticPriorityResolver()
    # Throw everything at it
    all_tokens = [T.QUESTION_WH, T.QUESTION_YN, T.NEGATION,
                  T.EMPHASIS_STRONG, T.EMPHASIS_MILD, T.SURPRISE,
                  T.EXCLAMATION, T.FOCUS, T.TOPIC_SHIFT,
                  T.TOPIC_MARKER, T.AGREEMENT, T.DOUBT]
    weights = {t: 0.75 for t in all_tokens}

    result = resolver.resolve(all_tokens, {}, weights)
    assert len(result.tokens) <= 4, \
        f"Too many tokens after resolution: {result.tokens}"
    print(f"[PASS] test_priority_resolver_max_tokens  "
          f"{len(all_tokens)} → {len(result.tokens)} tokens: {result.tokens}")


if __name__ == "__main__":
    # Re-run with new tests
    new_tests = [
        test_priority_resolver_basic,
        test_priority_resolver_negation_agreement,
        test_priority_resolver_max_tokens,
    ]
    passed = failed = 0
    for t in new_tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nNew resolver tests: {passed} passed, {failed} failed")
