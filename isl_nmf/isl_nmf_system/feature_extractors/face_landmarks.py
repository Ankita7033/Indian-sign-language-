"""
feature_extractors/face_landmarks.py
=====================================
Initialises MediaPipe FaceMesh and Pose detectors and provides
a unified per-frame landmark extraction interface used by all
downstream feature extractors.

Supports both:
  - mediapipe < 0.10.x  (.solutions API)
  - mediapipe >= 0.10.30 (new Tasks API, Python 3.11+/3.13)

Returns a FaceLandmarkResult dataclass containing:
  - normalised + pixel landmarks for face mesh (468 + optional iris)
  - pose landmarks (33 body keypoints)
  - face bounding-box metadata used for normalisation
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, Tuple

from config.config import SystemConfig, DEFAULT_CONFIG
from utils.math_utils import compute_face_bbox
from utils.logger import get_logger

log = get_logger(__name__)

# ── Detect which mediapipe API is available ──────────────────────────────────
_HAS_SOLUTIONS = hasattr(mp, "solutions")

if not _HAS_SOLUTIONS:
    # mediapipe 0.10.30+ Tasks API
    try:
        import mediapipe.tasks
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions
        _HAS_TASKS = True
        log.info("Using mediapipe Tasks API (0.10.30+)")
    except Exception as _e:
        _HAS_TASKS = False
        log.warning(f"Tasks API also unavailable: {_e}. Running in no-face mode.")
else:
    _HAS_TASKS = False
    log.info("Using mediapipe .solutions API (legacy)")


@dataclass
class FaceLandmarkResult:
    """Unified per-frame landmark container."""
    face_detected: bool = False
    pose_detected: bool = False
    num_faces: int = 0

    # Raw MediaPipe landmark objects
    face_landmarks: Optional[object]  = None
    pose_landmarks: Optional[object]  = None

    # Convenience pixel arrays
    face_pts_px: Optional[np.ndarray] = None   # shape (468, 3)
    iris_left_px: Optional[np.ndarray] = None  # shape (5, 3)
    iris_right_px: Optional[np.ndarray] = None # shape (5, 3)

    # Frame metadata
    img_w: int = 0
    img_h: int = 0

    # Face bounding box (pixels)
    face_x_min: float = 0.0
    face_y_min: float = 0.0
    face_x_max: float = 0.0
    face_y_max: float = 0.0
    face_width:  float = 1.0
    face_height: float = 1.0

    # Inter-pupillary distance (normalisation baseline)
    ipd_px: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build a FaceLandmarkResult from the new Tasks API output
# ─────────────────────────────────────────────────────────────────────────────

def _build_result_from_tasks(detection_result, w: int, h: int) -> FaceLandmarkResult:
    """Convert a mediapipe Tasks FaceLandmarker result into FaceLandmarkResult."""
    result = FaceLandmarkResult(img_w=w, img_h=h)

    if not detection_result or not detection_result.face_landmarks:
        return result

    result.num_faces = len(detection_result.face_landmarks)
    result.face_detected = True
    lms = detection_result.face_landmarks[0]   # first face

    n_lm = len(lms)
    pts = np.zeros((n_lm, 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        pts[i] = [lm.x * w, lm.y * h, lm.z * w]
    result.face_pts_px = pts
    result.face_landmarks = lms   # keep raw for compatibility

    if n_lm > 468:
        result.iris_left_px  = pts[468:473]
        result.iris_right_px = pts[473:478]

    # BBox from pts
    xs, ys = pts[:, 0], pts[:, 1]
    result.face_x_min  = float(xs.min())
    result.face_y_min  = float(ys.min())
    result.face_x_max  = float(xs.max())
    result.face_y_max  = float(ys.max())
    result.face_width  = float(result.face_x_max - result.face_x_min)
    result.face_height = float(result.face_y_max - result.face_y_min)

    left_eye_cx  = pts[468][0] if n_lm > 468 else pts[263][0]
    right_eye_cx = pts[473][0] if n_lm > 473 else pts[33][0]
    result.ipd_px = max(abs(right_eye_cx - left_eye_cx), 1.0)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tasks-based extractor (mediapipe 0.10.30+)
# ─────────────────────────────────────────────────────────────────────────────

class _TasksExtractor:
    """Uses mediapipe Tasks API (FaceLandmarker + PoseLandmarker)."""

    def __init__(self, config: SystemConfig):
        self.cfg = config
        import urllib.request, os, tempfile

        # Download model bundles if missing
        self._face_model = self._get_model(
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        self._pose_model = self._get_model(
            "pose_landmarker_full.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )

        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions

        with open(self._face_model, "rb") as f:
            face_model_data = f.read()
        with open(self._pose_model, "rb") as f:
            pose_model_data = f.read()

        face_opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=face_model_data),
            num_faces=config.mediapipe.face_mesh_max_faces,
            min_face_detection_confidence=config.mediapipe.face_mesh_min_detection_conf,
            min_face_presence_confidence=config.mediapipe.face_mesh_min_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_lm = FaceLandmarker.create_from_options(face_opts)

        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=pose_model_data),
            num_poses=1,
            min_pose_detection_confidence=config.mediapipe.pose_min_detection_conf,
            min_pose_presence_confidence=config.mediapipe.pose_min_tracking_conf,
        )
        self.pose_lm = PoseLandmarker.create_from_options(pose_opts)
        log.info("Tasks-based FaceLandmarkExtractor initialised.")

    def _get_model(self, filename: str, url: str) -> str:
        import os, urllib.request
        model_dir = os.path.join(os.path.dirname(__file__), "..", ".mp_models")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            log.info(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, path)
            log.info(f"Downloaded {filename}.")
        return path

    def process(self, bgr_frame: np.ndarray) -> FaceLandmarkResult:
        import mediapipe as mp
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_lm.detect(mp_image)
        pose_result = self.pose_lm.detect(mp_image)

        result = _build_result_from_tasks(face_result, w, h)
        result.img_w = w
        result.img_h = h

        if pose_result and pose_result.pose_landmarks:
            result.pose_detected = True
            result.pose_landmarks = pose_result.pose_landmarks[0]

        return result

    def close(self):
        self.face_lm.close()
        self.pose_lm.close()


# ─────────────────────────────────────────────────────────────────────────────
# Solutions-based extractor (mediapipe < 0.10.x, legacy)
# ─────────────────────────────────────────────────────────────────────────────

class _SolutionsExtractor:
    """Uses legacy mediapipe .solutions API."""

    def __init__(self, config: SystemConfig):
        self.cfg = config
        mp_fm   = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose

        self.face_mesh = mp_fm.FaceMesh(
            max_num_faces=config.mediapipe.face_mesh_max_faces,
            refine_landmarks=config.mediapipe.face_mesh_refine_landmarks,
            min_detection_confidence=config.mediapipe.face_mesh_min_detection_conf,
            min_tracking_confidence=config.mediapipe.face_mesh_min_tracking_conf,
        )
        self.pose = mp_pose.Pose(
            min_detection_confidence=config.mediapipe.pose_min_detection_conf,
            min_tracking_confidence=config.mediapipe.pose_min_tracking_conf,
            model_complexity=1,
        )
        log.info("Solutions FaceLandmarkExtractor initialised (FaceMesh + Pose).")

    def process(self, bgr_frame: np.ndarray) -> FaceLandmarkResult:
        result = FaceLandmarkResult()
        h, w = bgr_frame.shape[:2]
        result.img_w = w
        result.img_h = h

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        fm_results   = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if fm_results.multi_face_landmarks:
            result.face_detected = True
            lms = fm_results.multi_face_landmarks[0]
            result.face_landmarks = lms

            n_lm = len(lms.landmark)
            pts = np.zeros((n_lm, 3), dtype=np.float32)
            for i, lm in enumerate(lms.landmark):
                pts[i] = [lm.x * w, lm.y * h, lm.z * w]
            result.face_pts_px = pts

            if n_lm > 468:
                result.iris_left_px  = pts[468:473]
                result.iris_right_px = pts[473:478]

            bbox = compute_face_bbox(lms, w, h)
            (result.face_x_min, result.face_y_min,
             result.face_x_max, result.face_y_max,
             result.face_width, result.face_height) = bbox

            left_eye_cx  = pts[468][0] if n_lm > 468 else pts[263][0]
            right_eye_cx = pts[473][0] if n_lm > 473 else pts[33][0]
            result.ipd_px = max(abs(right_eye_cx - left_eye_cx), 1.0)

        if pose_results.pose_landmarks:
            result.pose_detected  = True
            result.pose_landmarks = pose_results.pose_landmarks

        return result

    def close(self):
        self.face_mesh.close()
        self.pose.close()


# ─────────────────────────────────────────────────────────────────────────────
# Null extractor — fallback when no mediapipe backend works
# ─────────────────────────────────────────────────────────────────────────────

class _NullExtractor:
    def process(self, bgr_frame: np.ndarray) -> FaceLandmarkResult:
        h, w = bgr_frame.shape[:2]
        r = FaceLandmarkResult()
        r.img_w, r.img_h = w, h
        return r
    def close(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Public facade — picks the right backend automatically
# ─────────────────────────────────────────────────────────────────────────────

class FaceLandmarkExtractor:
    """
    Wraps MediaPipe FaceMesh + Pose in a single object.
    Automatically selects the correct API for the installed mediapipe version.

    Usage
    -----
    extractor = FaceLandmarkExtractor(config)
    result = extractor.process(bgr_frame)
    # result.face_pts_px  -> (468, 3) pixel coords
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        if _HAS_SOLUTIONS:
            self._impl = _SolutionsExtractor(config)
        elif _HAS_TASKS:
            self._impl = _TasksExtractor(config)
        else:
            log.warning("No mediapipe backend available. Running without face detection.")
            self._impl = _NullExtractor()

    def process(self, bgr_frame: np.ndarray) -> FaceLandmarkResult:
        return self._impl.process(bgr_frame)

    def close(self):
        self._impl.close()
        log.info("FaceLandmarkExtractor closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
