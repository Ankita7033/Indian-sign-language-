"""
utils/math_utils.py — Geometric and statistical helper functions.
"""
import numpy as np
from typing import List, Tuple


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns angle in degrees between two vectors."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def normalise_by_face_width(value: float, face_width: float) -> float:
    """Normalise a pixel measurement by the face bounding box width."""
    if face_width < 1e-6:
        return 0.0
    return value / face_width


def moving_average(buffer: List[float], new_val: float, window: int) -> Tuple[List[float], float]:
    buffer.append(new_val)
    if len(buffer) > window:
        buffer.pop(0)
    return buffer, float(np.mean(buffer))


def ema_filter(prev: float, new_val: float, alpha: float) -> float:
    """Exponential moving average: alpha in (0,1); lower = smoother."""
    return alpha * new_val + (1.0 - alpha) * prev


def landmark_to_np(landmark, img_w: int, img_h: int) -> np.ndarray:
    """Convert a single MediaPipe NormalisedLandmark to pixel coords."""
    return np.array([landmark.x * img_w, landmark.y * img_h, landmark.z * img_w])


def landmarks_to_np_array(landmarks, indices: List[int],
                           img_w: int, img_h: int) -> np.ndarray:
    """Extract a subset of face mesh landmarks as (N,3) pixel-coord array."""
    pts = []
    lm_list = list(landmarks.landmark)
    for idx in indices:
        lm = lm_list[idx]
        pts.append([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    return np.array(pts, dtype=np.float32)


def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - np.max(arr))
    return e / e.sum()


def compute_face_bbox(face_landmarks, img_w: int, img_h: int):
    """Return (x_min, y_min, x_max, y_max, width, height) in pixels."""
    xs = [lm.x * img_w for lm in face_landmarks.landmark]
    ys = [lm.y * img_h for lm in face_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max, x_max - x_min, y_max - y_min
