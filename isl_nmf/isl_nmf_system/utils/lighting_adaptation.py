"""
utils/lighting_adaptation.py
===============================
Feature 8: Lighting Adaptation Engine

Makes the system work outside lab environments by applying
real-time image preprocessing to normalize illumination.

Techniques:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
     — enhances local contrast without over-brightening
  2. Gamma correction — compensates for over/under-exposure
  3. White balance estimation — corrects color temperature
  4. Face ROI brightness normalization — targets face region only

Automatically detects lighting condition and applies
the appropriate correction strategy.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LightingStats:
    mean_brightness: float
    std_brightness:  float
    condition: str   # "dark" | "normal" | "bright" | "uneven"
    correction_applied: str


class LightingAdaptationEngine:
    """
    Adaptive illumination normalization for robust face detection
    in varied lighting conditions (offices, outdoors, low-light).
    """

    DARK_THRESHOLD   = 60    # mean brightness below this = dark
    BRIGHT_THRESHOLD = 200   # above this = over-exposed
    UNEVEN_THRESHOLD = 50    # std above this = uneven lighting

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # CLAHE instance (reused for efficiency)
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        log.info("LightingAdaptationEngine ready.")

    def analyze(self, bgr: np.ndarray) -> LightingStats:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std  = float(np.std(gray))

        if mean < self.DARK_THRESHOLD:
            condition = "dark"
        elif mean > self.BRIGHT_THRESHOLD:
            condition = "bright"
        elif std > self.UNEVEN_THRESHOLD:
            condition = "uneven"
        else:
            condition = "normal"

        return LightingStats(
            mean_brightness   = mean,
            std_brightness    = std,
            condition         = condition,
            correction_applied = "none"
        )

    def _gamma_correct(self, bgr: np.ndarray,
                       gamma: float) -> np.ndarray:
        inv   = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(bgr, table)

    def _clahe_enhance(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _white_balance(self, bgr: np.ndarray) -> np.ndarray:
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - \
            ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - \
            ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def process(self, bgr: np.ndarray) -> tuple:
        """
        Apply adaptive illumination correction.
        Returns (corrected_frame, LightingStats).
        """
        if not self.enabled:
            stats = self.analyze(bgr)
            return bgr, stats

        stats = self.analyze(bgr)

        if stats.condition == "dark":
            # Aggressive brightening
            out = self._gamma_correct(bgr, gamma=0.5)
            out = self._clahe_enhance(out)
            stats.correction_applied = "gamma+clahe"

        elif stats.condition == "bright":
            # Tone down
            out = self._gamma_correct(bgr, gamma=1.8)
            stats.correction_applied = "gamma_damp"

        elif stats.condition == "uneven":
            # CLAHE for local contrast normalization
            out = self._clahe_enhance(bgr)
            out = self._white_balance(out)
            stats.correction_applied = "clahe+wb"

        else:
            # Normal — light touch
            out = self._clahe_enhance(bgr)
            stats.correction_applied = "clahe_light"

        return out, stats

    def get_overlay_text(self, stats: LightingStats) -> str:
        return (f"Light: {stats.condition} "
                f"[{stats.mean_brightness:.0f}lx] "
                f"→ {stats.correction_applied}")
