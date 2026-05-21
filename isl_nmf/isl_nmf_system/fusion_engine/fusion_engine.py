"""
fusion_engine/fusion_engine.py
================================
The Fusion Engine is the central coordinator of the ISL NMF pipeline.

It:
  1. Receives per-frame BGR image
  2. Dispatches to all feature extractors in order
  3. Packages results into a normalised FeatureVector (Dict[str, float])
  4. Feeds the FeatureVector into the Semantic Fusion Graph
  5. Returns a FusionResult with full feature breakdown + semantic tokens

FeatureVector key conventions:
  All values are in [0, 1] unless otherwise noted.
  Boolean features are encoded as 1.0 (True) or 0.0 (False).
  Directional features use signed floats (e.g., head_yaw in degrees).
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.config import SystemConfig, DEFAULT_CONFIG
from feature_extractors.face_landmarks   import FaceLandmarkExtractor, FaceLandmarkResult
from feature_extractors.head_pose        import HeadPoseEstimator, HeadPoseFeatures
from feature_extractors.eyebrow_tracker  import EyebrowTracker, EyebrowFeatures
from feature_extractors.eye_tracking     import EyeTracker, EyeTrackingFeatures
from feature_extractors.lip_contour      import LipContourExtractor, LipContourFeatures
from feature_extractors.shoulder_tracker import ShoulderTracker, ShoulderFeatures
from feature_extractors.optical_flow_tracker import OpticalFlowTracker, OpticalFlowFeatures
from feature_extractors.temporal_smoother    import TemporalSmoother
from feature_extractors.hand_gesture         import HandGestureDetector, HandFeatures
from semantic_graph.semantic_graph_builder   import SemanticFusionGraph, SemanticGraphState
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FusionResult:
    """Complete per-frame output of the ISL NMF pipeline."""
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    process_time_ms: float = 0.0

    # Raw feature structs
    landmark_result:  Optional[FaceLandmarkResult]     = None
    head_pose:        Optional[HeadPoseFeatures]       = None
    eyebrow:          Optional[EyebrowFeatures]        = None
    eye:              Optional[EyeTrackingFeatures]    = None
    lip:              Optional[LipContourFeatures]     = None
    shoulder:         Optional[ShoulderFeatures]       = None
    optical_flow:     Optional[OpticalFlowFeatures]    = None
    hand:             Optional[HandFeatures]           = None
    hand_gesture:     str                              = "none"

    # Semantic graph output
    graph_state:      Optional[SemanticGraphState]     = None

    # Final output string
    semantic_output: str = "NEUTRAL"

    # Normalised feature vector fed to graph
    feature_vector: Dict[str, float] = field(default_factory=dict)


def _bool_to_float(b: bool) -> float:
    return 1.0 if b else 0.0


def _sigmoid_deg(deg: float, scale: float = 30.0) -> float:
    """Map degrees to [0,1] via sigmoid for smooth evidence encoding."""
    import math
    return 1.0 / (1.0 + math.exp(-abs(deg) / scale))


class FusionEngine:
    """
    Orchestrates all feature extractors and the semantic fusion graph.

    Usage
    -----
    engine = FusionEngine(config)
    result = engine.process_frame(bgr_frame, frame_idx=i)
    print(result.semantic_output)
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        log.info("Initialising FusionEngine...")

        self.lm_extractor   = FaceLandmarkExtractor(config)
        self.head_pose_est  = HeadPoseEstimator(config)
        self.eyebrow_track  = EyebrowTracker(config)
        self.eye_tracker    = EyeTracker(config)
        self.lip_extractor  = LipContourExtractor(config)
        self.shoulder_track = ShoulderTracker(config)
        self.flow_tracker   = OpticalFlowTracker(config)
        self.smoother       = TemporalSmoother(config)
        self.sfg            = SemanticFusionGraph(config)
        self.hand_detector  = HandGestureDetector()

        self._frame_count = 0
        log.info("FusionEngine ready.")

    def process_frame(self, bgr_frame: np.ndarray,
                      frame_idx: int = -1) -> FusionResult:
        t0 = time.perf_counter()
        self._frame_count += 1
        fi = frame_idx if frame_idx >= 0 else self._frame_count

        result = FusionResult(
            frame_idx    = fi,
            timestamp_ms = time.time() * 1000
        )

        # ---- Step 1: Landmark extraction ----
        lm_res = self.lm_extractor.process(bgr_frame)
        result.landmark_result = lm_res

        # ---- Step 2: Feature extraction ----
        head   = self.head_pose_est.process(lm_res)
        brow   = self.eyebrow_track.process(lm_res)
        eye    = self.eye_tracker.process(lm_res)
        lip    = self.lip_extractor.process(lm_res)
        sho    = self.shoulder_track.process(lm_res)
        flow   = self.flow_tracker.process(bgr_frame)
        hand   = self.hand_detector.process(bgr_frame)

        result.head_pose     = head
        result.eyebrow       = brow
        result.eye           = eye
        result.lip           = lip
        result.shoulder      = sho
        result.optical_flow  = flow
        result.hand          = hand
        result.hand_gesture  = hand.combined_gesture

        # ---- Step 3: Build FeatureVector ----
        fv = self._build_feature_vector(head, brow, eye, lip, sho, flow)
        result.feature_vector = fv

        # ---- Step 4: Temporal smoothing ----
        scalar_map = {
            "left_brow_height":        brow.left_brow_height,
            "right_brow_height":       brow.right_brow_height,
            "interbrow_distance":      brow.interbrow_distance,
            "mean_ear":                eye.mean_ear,
            "left_ear":                eye.left_ear,
            "right_ear":               eye.right_ear,
            "gaze_x":                  eye.gaze_x,
            "gaze_y":                  eye.gaze_y,
            "mar":                     lip.mar,
            "lip_open":                lip.lip_open,
            "lip_spread":              lip.lip_spread,
            "lip_protrusion":          lip.lip_protrusion,
            "shoulder_bilateral_raise": sho.bilateral_raise,
            "shoulder_lateral_lean":   sho.lateral_lean,
            "shoulder_unilateral_shrug": sho.unilateral_shrug,
            "head_pitch":              head.pitch_deg,
            "head_yaw":                head.yaw_deg,
            "head_roll":               head.roll_deg,
            "flow_global_mag":         flow.global_magnitude,
        }
        discrete_map = {
            "both_raised":   brow.both_raised,
            "furrowed":      brow.furrowed,
            "mouth_open":    lip.mouth_open,
            "is_shrugging":  sho.is_shrugging,
            "is_nodding":    head.is_nodding,
            "is_shaking":    head.is_shaking,
        }
        self.smoother.update(scalar_map, discrete_map)

        # ---- Step 5: Semantic graph update ----
        graph_state = self.sfg.update(fv)
        result.graph_state = graph_state

        # ---- Step 6: Format output ----
        result.semantic_output = self._format_output(graph_state)

        result.process_time_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    def _build_feature_vector(self,
                               head:  HeadPoseFeatures,
                               brow:  EyebrowFeatures,
                               eye:   EyeTrackingFeatures,
                               lip:   LipContourFeatures,
                               sho:   ShoulderFeatures,
                               flow:  OpticalFlowFeatures) -> Dict[str, float]:
        """
        Map raw feature structs to a normalised [0,1] dict
        used as evidence for the Semantic Fusion Graph.
        """
        thr = self.cfg.thresholds
        fv: Dict[str, float] = {}

        # ---- Eyebrow features ----
        # Normalise raise height: map [0, 0.08] -> [0, 1]
        fv["left_brow_raise"]  = min(brow.left_brow_height  / 0.08, 1.0)
        fv["right_brow_raise"] = min(brow.right_brow_height / 0.08, 1.0)
        # both_raised: require brows to be SIGNIFICANTLY above baseline, not just above threshold
        fv["both_raised"]      = min(max(0.0, (brow.mean_brow_height - 0.055) / 0.035), 1.0)
        fv["brow_raise_one"]   = _bool_to_float(brow.left_raised ^ brow.right_raised)
        fv["furrowed"]         = _bool_to_float(brow.furrowed)
        fv["brow_velocity"]    = min(brow.brow_velocity / 0.06, 1.0)  # higher divisor = less sensitive

        # ---- Eye features ----
        fv["mean_ear"]         = min(eye.mean_ear / 0.40, 1.0)
        fv["wide_eye"]         = _bool_to_float(eye.is_wide_open)
        fv["blink"]            = _bool_to_float(eye.is_blinking)
        fv["gaze_forward"]     = max(0.0, 1.0 - abs(eye.gaze_x) / 0.20
                                         - abs(eye.gaze_y) / 0.15)  # tighter: neutral gaze scores lower
        fv["gaze_lateral"]     = min(abs(eye.gaze_x) / 0.30, 1.0)
        fv["gaze_up"]          = min(max(0.0, -eye.gaze_y / 0.20), 1.0) if eye.gaze_y < 0 else 0.0  # capped
        fv["gaze_down"]        = min(max(0.0, eye.gaze_y / 0.20), 1.0) if eye.gaze_y > 0 else 0.0  # capped

        # ---- Head pose ----
        # pitch: positive = nod down
        fv["head_nod"]      = min(abs(head.pitch_deg) / 35.0, 1.0) if head.is_nodding else 0.0  # linear, not sigmoid
        fv["head_pitch_up"] = max(0.0, -head.pitch_deg / 30.0) if not head.is_nodding else 0.0
        fv["is_shaking"]    = _bool_to_float(head.is_shaking)
        fv["head_tilt"]     = _bool_to_float(head.is_tilting)
        fv["head_valid"]    = _bool_to_float(head.valid)

        # ---- Lip features ----
        fv["mouth_open"]    = _bool_to_float(lip.mouth_open)
        fv["lip_spread"]    = min(lip.lip_spread / 0.10, 1.0)
        fv["lip_rounded"]   = _bool_to_float(lip.lip_rounded)
        fv["lip_pursed"]    = _bool_to_float(lip.lip_pursed)
        fv["lip_protrusion"]= min(lip.lip_protrusion / 0.05, 1.0)

        # ---- Shoulder features ----
        fv["shoulder_bilateral_raise"] = min(max(sho.bilateral_raise / 0.08, 0.0), 1.0)
        fv["shoulder_lateral_lean"]    = min(abs(sho.lateral_lean)   / 0.06, 1.0)
        fv["is_shrugging"]             = _bool_to_float(sho.is_shrugging)

        # ---- Optical flow ----
        fv["flow_active"]   = _bool_to_float(flow.global_magnitude > thr.flow_magnitude_threshold)

        # ---- Stability proxy ----
        # "face_stable" = low overall motion
        fv["face_stable"] = max(0.0, 1.0 - flow.global_magnitude / 3.0)  # stricter stability check

        return fv

    def _format_output(self, state: SemanticGraphState) -> str:
        """Convert active graph nodes to the canonical output string."""
        tokens = state.token_sequence
        if not tokens:
            return "NEUTRAL"
        return " ".join(tokens)

    def close(self):
        self.lm_extractor.close()
        log.info("FusionEngine closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
