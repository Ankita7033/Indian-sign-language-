"""
config.py
=========
Central configuration for the ISL Non-Manual Feature Extraction System.
"""

from dataclasses import dataclass, field


class LandmarkIndices:
    LEFT_EYEBROW_UPPER  = [336, 296, 334, 293, 300]
    LEFT_EYEBROW_LOWER  = [285, 295, 282, 283, 276]
    RIGHT_EYEBROW_UPPER = [70,  63,  105, 66,  107]
    RIGHT_EYEBROW_LOWER = [46,  53,  52,  65,  55]
    LEFT_EYE_UPPER  = [386, 385, 384, 381]
    LEFT_EYE_LOWER  = [380, 374, 373, 390]
    RIGHT_EYE_UPPER = [159, 158, 157, 154]
    RIGHT_EYE_LOWER = [145, 153, 144, 163]
    LEFT_EAR_LANDMARKS  = [362, 385, 387, 263, 373, 380]
    RIGHT_EAR_LANDMARKS = [33,  160, 158, 133, 153, 144]
    LEFT_IRIS_CENTER  = 468
    RIGHT_IRIS_CENTER = 473
    OUTER_LIP_UPPER = [61, 185, 40, 39, 37, 0,  267, 269, 270, 409, 291]
    OUTER_LIP_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    INNER_LIP_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    INNER_LIP_LOWER = [78, 95,  88, 178, 87, 14, 317, 402, 318, 324, 308]
    LIP_LEFT_CORNER  = 61
    LIP_RIGHT_CORNER = 291
    LIP_TOP    = 13
    LIP_BOTTOM = 14
    NOSE_TIP    = 1
    NOSE_BRIDGE = 6
    CHIN = 152
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                 323, 361, 288, 397, 365, 379, 378, 400, 377,
                 152, 148, 176, 149, 150, 136, 172, 58, 132,
                 93,  234, 127, 162, 21,  54,  103, 67,  109]
    POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


FACE_3D_MODEL_POINTS = [
    (0.0,    0.0,    0.0   ),
    (0.0,   -330.0, -65.0  ),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
]


@dataclass
class ThresholdConfig:
    eyebrow_raise_threshold: float     = 0.072   # very high — only obvious raises
    eyebrow_furrow_threshold: float    = 0.010
    eyebrow_asymmetry_threshold: float = 0.018
    ear_blink_threshold: float         = 0.19
    ear_wide_threshold: float          = 0.42    # only truly wide eyes
    gaze_horizontal_threshold: float   = 0.35    # ignore drift
    gaze_vertical_threshold: float     = 0.30
    head_nod_threshold: float          = 22.0    # only strong deliberate nod
    head_shake_threshold: float        = 25.0    # only strong deliberate shake
    head_tilt_threshold: float         = 18.0
    lip_open_threshold: float          = 0.060   # only clearly open mouth
    lip_spread_threshold: float        = 0.70
    lip_protrusion_threshold: float    = 0.025
    shoulder_raise_threshold: float    = 0.070   # only obvious shrug
    shoulder_shrug_threshold: float    = 0.090
    shoulder_lean_threshold: float     = 0.055
    flow_magnitude_threshold: float    = 4.5     # ignore micro-movements
    flow_region_size: int              = 32
    smoother_window: int               = 12      # heavy smoothing
    smoother_alpha: float              = 0.18    # slow response = stable output
    graph_activation_threshold: float  = 0.75    # very high bar to fire a token
    graph_decay_rate: float            = 0.22    # tokens clear quickly


@dataclass
class CameraConfig:
    device_id: int         = 0
    frame_width: int       = 1280
    frame_height: int      = 720
    target_fps: int        = 30
    flip_horizontal: bool  = True


@dataclass
class MediaPipeConfig:
    face_mesh_max_faces: int            = 4
    face_mesh_refine_landmarks: bool    = True
    face_mesh_min_detection_conf: float = 0.6
    face_mesh_min_tracking_conf: float  = 0.6
    pose_min_detection_conf: float      = 0.5
    pose_min_tracking_conf: float       = 0.5
    use_holistic: bool                  = False


class LinguisticTokens:
    QUESTION_WH    = "QUESTION(type=WH)"
    QUESTION_YN    = "QUESTION(type=YN)"
    NEGATION       = "NEGATION(active)"
    ASSERTION      = "ASSERTION"
    EMPHASIS_STRONG = "EMPHASIS(strong)"
    EMPHASIS_MILD  = "EMPHASIS(mild)"
    TOPIC_SHIFT    = "TOPIC_SHIFT(true)"
    CONDITIONAL    = "CONDITIONAL"
    EXCLAMATION    = "EXCLAMATION"
    DOUBT          = "DOUBT"
    SURPRISE       = "SURPRISE"
    AGREEMENT      = "AGREEMENT"
    DISAGREEMENT   = "DISAGREEMENT"
    UNCERTAINTY    = "UNCERTAINTY"
    CONFIRMATION   = "CONFIRMATION"
    TOPIC_MARKER   = "TOPIC_MARKER"
    FOCUS          = "FOCUS"
    BOUNDARY       = "SENTENCE_BOUNDARY"
    NEUTRAL        = "NEUTRAL"


OPTICAL_FLOW_ROIS = {
    "forehead":    (0.25, 0.05, 0.50, 0.15),
    "nose":        (0.35, 0.35, 0.30, 0.20),
    "left_cheek":  (0.05, 0.30, 0.25, 0.25),
    "right_cheek": (0.70, 0.30, 0.25, 0.25),
    "mouth":       (0.25, 0.60, 0.50, 0.20),
    "chin":        (0.30, 0.78, 0.40, 0.15),
}


class PoseLandmarkIndices:
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_EAR       = 7
    RIGHT_EAR      = 8
    NOSE           = 0
    LEFT_HIP       = 23
    RIGHT_HIP      = 24


@dataclass
class EvaluationConfig:
    output_dir: str               = "evaluation/results"
    log_latency: bool             = True
    latency_window: int           = 100
    semantic_align_weight: float  = 0.7
    save_frames: bool             = False


@dataclass
class SystemConfig:
    thresholds: ThresholdConfig  = field(default_factory=ThresholdConfig)
    camera: CameraConfig         = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig   = field(default_factory=MediaPipeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    debug_mode: bool             = False
    show_visualization: bool     = True
    log_to_file: bool            = False
    log_path: str                = "logs/isl_nmf.log"


DEFAULT_CONFIG = SystemConfig()
