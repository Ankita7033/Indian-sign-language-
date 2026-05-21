"""
feature_extractors/hand_gesture.py
=====================================
Hand Gesture Detection using MediaPipe Hands.

Detects and classifies basic hand shapes used in ISL:
  - Open palm
  - Closed fist
  - Pointing index
  - Thumbs up / down
  - Victory / peace sign
  - OK sign
  - Pinch

These hand shape labels are fused with non-manual grammar tokens
in the FusionEngine to produce complete ISL interpretations like:
  GO (hand sign) + QUESTION(type=WH) (eyebrow raise) = "Where are you going?"

Finger state is determined by comparing fingertip Y positions
against knuckle Y positions (classical geometric approach, no ML).
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HandFeatures:
    # Per hand
    left_detected:  bool = False
    right_detected: bool = False

    left_gesture:  str = "none"    # gesture label
    right_gesture: str = "none"

    left_confidence:  float = 0.0
    right_confidence: float = 0.0

    # Finger extension states (True = extended)
    left_fingers:  List[bool] = field(default_factory=lambda: [False]*5)
    right_fingers: List[bool] = field(default_factory=lambda: [False]*5)
    # [thumb, index, middle, ring, pinky]

    # Hand position (normalised 0-1)
    left_wrist_pos:  Optional[Tuple[float,float]] = None
    right_wrist_pos: Optional[Tuple[float,float]] = None

    # Combined gesture for fusion
    combined_gesture: str = "none"


# Finger landmark indices in MediaPipe Hands (21 points)
# Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]   # second joint
FINGER_MCPS = [2, 5,  9, 13, 17]   # knuckle


def _dist(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def _fingers_extended(landmarks, handedness: str) -> List[bool]:
    """
    Determine which fingers are extended using Euclidean distance to the wrist.
    This makes the detection invariant to camera mirroring and hand rotation.
    """
    lm = landmarks if isinstance(landmarks, list) else getattr(landmarks, 'landmark', landmarks)
    extended = []
    wrist = lm[0]

    # Thumb
    extended.append(_dist(lm[4], wrist) > _dist(lm[3], wrist))

    # Index, middle, ring, pinky
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        extended.append(_dist(lm[tip], wrist) > _dist(lm[pip], wrist))

    return extended


def _classify_gesture(fingers: List[bool]) -> str:
    """
    Rule-based gesture classification from finger extension pattern.
    fingers = [thumb, index, middle, ring, pinky]
    """
    t, i, m, r, p = fingers

    # All extended = open palm
    if all(fingers):
        return "open_palm"

    # All closed = fist
    if not any(fingers):
        return "fist"

    # Only index extended = pointing
    if i and not m and not r and not p:
        return "point"

    # Index + middle extended = victory/peace
    if i and m and not r and not p:
        return "victory"

    # Thumb + index form circle (OK) — approximation
    if t and not i and not m and not r and p:
        return "thumbs_up"

    # Thumb down (thumb + all others closed, thumb pointing down)
    if not i and not m and not r and not p and t:
        return "thumb"

    # Index + middle + ring = three fingers
    if not t and i and m and r and not p:
        return "three_fingers"

    # Pinky only
    if not t and not i and not m and not r and p:
        return "pinky"

    return "custom"


class HandGestureDetector:
    """
    Detects hand gestures from webcam frames using MediaPipe Hands.
    Runs alongside the face pipeline for full ISL interpretation.
    """

    def __init__(self, max_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.6):
        import urllib.request, os
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions
        
        # Download model bundle if missing
        model_dir = os.path.join(os.path.dirname(__file__), "..", ".mp_models")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(path):
            log.info("Downloading hand_landmarker.task ...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, path)
            log.info("Downloaded hand_landmarker.task.")
            
        with open(path, "rb") as f:
            model_data = f.read()

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.hands = HandLandmarker.create_from_options(opts)
        log.info("HandGestureDetector ready.")

    def process(self, bgr_frame: np.ndarray) -> HandFeatures:
        import mediapipe as mp
        feat = HandFeatures()
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hands.detect(mp_img)

        if not getattr(results, 'hand_landmarks', []):
            return feat

        for hand_lms, handedness in zip(
            results.hand_landmarks,
            results.handedness
        ):
            # handedness is a list of Category objects
            label      = handedness[0].category_name   # "Left" or "Right"
            score      = handedness[0].score
            fingers    = _fingers_extended(hand_lms, label)
            gesture    = _classify_gesture(fingers)
            wrist      = hand_lms[0]
            wrist_pos  = (wrist.x, wrist.y)

            if label == "Left":
                feat.left_detected   = True
                feat.left_gesture    = gesture
                feat.left_confidence = score
                feat.left_fingers    = fingers
                feat.left_wrist_pos  = wrist_pos
            else:
                feat.right_detected   = True
                feat.right_gesture    = gesture
                feat.right_confidence = score
                feat.right_fingers    = fingers
                feat.right_wrist_pos  = wrist_pos

        # Combined gesture
        if feat.left_detected and feat.right_detected:
            feat.combined_gesture = f"{feat.right_gesture}+{feat.left_gesture}"
        elif feat.right_detected:
            feat.combined_gesture = feat.right_gesture
        elif feat.left_detected:
            feat.combined_gesture = feat.left_gesture

        return feat

    def close(self):
        self.hands.close()
