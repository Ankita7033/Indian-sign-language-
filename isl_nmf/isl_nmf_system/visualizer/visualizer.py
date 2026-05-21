"""
visualizer/visualizer.py
==========================
Real-time OpenCV visualization overlay for the ISL NMF pipeline.

Renders on the live webcam frame:
  - Face mesh landmarks (selected subsets)
  - Eyebrow contours with raise indicator
  - Eye EAR bars
  - Iris gaze vector
  - Lip contour polygon
  - Head pose axes (3D projected arrows)
  - Shoulder keypoints
  - Optical flow vectors (ROI-level arrows)
  - Semantic graph activation bars (right panel)
  - Final linguistic token output (bottom banner)
  - FPS counter
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

from config.config import (
    SystemConfig, DEFAULT_CONFIG, LandmarkIndices, LinguisticTokens
)
from fusion_engine.fusion_engine import FusionResult
from semantic_graph.semantic_graph_builder import SemanticGraphState

_LI = LandmarkIndices
T   = LinguisticTokens

# ---- Color palette (BGR) ----
C_GREEN    = (50,  220,  50)
C_RED      = (50,   50, 220)
C_BLUE     = (220, 100,  50)
C_YELLOW   = (30,  220, 220)
C_CYAN     = (220, 200,  30)
C_WHITE    = (240, 240, 240)
C_GRAY     = (120, 120, 120)
C_ORANGE   = (30,  140, 255)
C_MAGENTA  = (220,  50, 220)
C_DARK     = (20,   20,  20)
C_PANEL    = (30,   30,  30)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


class Visualizer:
    """
    Draws all feature overlays and semantic output onto BGR frames.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg  = config
        self._fps_buf: List[float] = []
        self._last_ts: float = 0.0
        import time
        self._t0 = time.perf_counter()

    # =====================================================================
    # Public interface
    # =====================================================================

    def draw(self, frame: np.ndarray, result: FusionResult) -> np.ndarray:
        """
        Draw all overlays onto a copy of frame.
        Returns annotated frame.
        """
        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        if result.landmark_result and result.landmark_result.face_detected:
            pts = result.landmark_result.face_pts_px
            self._draw_eyebrow_contours(canvas, pts, result)
            self._draw_eye_indicators(canvas, pts, result)
            self._draw_lip_contour(canvas, result)
            self._draw_iris_gaze(canvas, result)

        if result.landmark_result and result.landmark_result.pose_detected:
            self._draw_shoulder_keypoints(canvas, result)

        if result.head_pose and result.head_pose.valid:
            self._draw_head_pose_axes(canvas, result)

        self._draw_optical_flow_rois(canvas, result)
        self._draw_graph_panel(canvas, result, w, h)
        self._draw_semantic_banner(canvas, result, w, h)
        self._draw_fps(canvas, result)

        return canvas

    # =====================================================================
    # Component renderers
    # =====================================================================

    def _draw_eyebrow_contours(self, canvas, pts, result: FusionResult):
        brow = result.eyebrow
        if brow is None:
            return
        for ids, color in [
            (_LI.LEFT_EYEBROW_UPPER  + _LI.LEFT_EYEBROW_LOWER,  C_YELLOW),
            (_LI.RIGHT_EYEBROW_UPPER + _LI.RIGHT_EYEBROW_LOWER, C_YELLOW),
        ]:
            poly = pts[ids, :2].astype(np.int32)
            cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=2)

        # Raise indicator bars
        lh = int(np.clip(brow.left_brow_height  / 0.08 * 30, 0, 30))
        rh = int(np.clip(brow.right_brow_height / 0.08 * 30, 0, 30))
        col = C_GREEN if brow.both_raised else (C_ORANGE if brow.furrowed else C_GRAY)
        cv2.rectangle(canvas, (5, 10), (15, 40), C_DARK, -1)
        cv2.rectangle(canvas, (5, 40-lh), (15, 40), col, -1)
        cv2.rectangle(canvas, (20, 10), (30, 40), C_DARK, -1)
        cv2.rectangle(canvas, (20, 40-rh), (30, 40), col, -1)
        cv2.putText(canvas, "BR", (3, 55), FONT, 0.35, col, 1)

    def _draw_eye_indicators(self, canvas, pts, result: FusionResult):
        eye = result.eye
        if eye is None:
            return
        # EAR vertical bars (left/right)
        lear = int(np.clip(eye.left_ear  / 0.40 * 30, 0, 30))
        rear = int(np.clip(eye.right_ear / 0.40 * 30, 0, 30))
        blink_col = C_RED if eye.is_blinking else (C_GREEN if eye.is_wide_open else C_GRAY)
        cv2.rectangle(canvas, (40, 10), (50, 40), C_DARK, -1)
        cv2.rectangle(canvas, (40, 40-lear), (50, 40), blink_col, -1)
        cv2.rectangle(canvas, (55, 10), (65, 40), C_DARK, -1)
        cv2.rectangle(canvas, (55, 40-rear), (65, 40), blink_col, -1)
        cv2.putText(canvas, "EAR", (38, 55), FONT, 0.35, blink_col, 1)

    def _draw_lip_contour(self, canvas, result: FusionResult):
        lip = result.lip
        if lip is None or lip.outer_lip_pts is None:
            return
        poly = lip.outer_lip_pts.astype(np.int32)
        col  = C_CYAN if lip.mouth_open else C_GRAY
        cv2.polylines(canvas, [poly], isClosed=True, color=col, thickness=2)
        if lip.inner_lip_pts is not None:
            ipoly = lip.inner_lip_pts.astype(np.int32)
            cv2.polylines(canvas, [ipoly], isClosed=True, color=C_MAGENTA, thickness=1)

    def _draw_iris_gaze(self, canvas, result: FusionResult):
        eye = result.eye
        lm  = result.landmark_result
        if eye is None or not eye.iris_available:
            return
        if lm.iris_left_px is not None:
            ic = lm.iris_left_px[0, :2].astype(int)
            cv2.circle(canvas, tuple(ic), 4, C_GREEN, -1)
            # Gaze vector arrow
            dx = int(eye.gaze_x * 30)
            dy = int(eye.gaze_y * 30)
            cv2.arrowedLine(canvas, tuple(ic),
                            (ic[0]+dx, ic[1]+dy), C_GREEN, 2, tipLength=0.4)
        if lm.iris_right_px is not None:
            ic = lm.iris_right_px[0, :2].astype(int)
            cv2.circle(canvas, tuple(ic), 4, C_GREEN, -1)
            dx = int(eye.gaze_x * 30)
            dy = int(eye.gaze_y * 30)
            cv2.arrowedLine(canvas, tuple(ic),
                            (ic[0]+dx, ic[1]+dy), C_GREEN, 2, tipLength=0.4)

    def _draw_shoulder_keypoints(self, canvas, result: FusionResult):
        sho = result.shoulder
        lm  = result.landmark_result
        if sho is None or lm.pose_landmarks is None:
            return
        plms = lm.pose_landmarks
        w, h = lm.img_w, lm.img_h
        for idx in [11, 12, 7, 8]:
            lm_ = plms.landmark[idx]
            px  = int(lm_.x * w)
            py  = int(lm_.y * h)
            col = C_ORANGE if sho.is_shrugging else C_BLUE
            cv2.circle(canvas, (px, py), 8, col, -1)
        # Shrug label
        if sho.is_shrugging:
            cv2.putText(canvas, "SHRUG", (10, 90), FONT, 0.5, C_ORANGE, 2)

    def _draw_head_pose_axes(self, canvas, result: FusionResult):
        head = result.head_pose
        lm   = result.landmark_result
        if head is None or head.rotation_vector is None:
            return
        pts = lm.face_pts_px
        # Nose tip as origin
        nose = pts[1, :2].astype(int)
        p  = head.pitch_deg
        y_ = head.yaw_deg
        r  = head.roll_deg

        # Simple 2D projection of pose arrows
        arrow_len = 50
        # Yaw -> horizontal arrow
        yaw_col = C_RED if head.is_shaking else C_WHITE
        cv2.arrowedLine(canvas, tuple(nose),
                        (int(nose[0] + np.sin(np.radians(y_)) * arrow_len),
                         int(nose[1])),
                        yaw_col, 2, tipLength=0.3)
        # Pitch -> vertical arrow
        pitch_col = C_GREEN if head.is_nodding else C_WHITE
        cv2.arrowedLine(canvas, tuple(nose),
                        (int(nose[0]),
                         int(nose[1] + np.sin(np.radians(p)) * arrow_len)),
                        pitch_col, 2, tipLength=0.3)
        # Text labels
        cv2.putText(canvas,
                    f"P:{p:+.1f} Y:{y_:+.1f} R:{r:+.1f}",
                    (10, canvas.shape[0] - 70), FONT, 0.45, C_WHITE, 1)

    def _draw_optical_flow_rois(self, canvas, result: FusionResult):
        flow = result.optical_flow
        if flow is None:
            return
        h, w = canvas.shape[:2]
        from config.config import OPTICAL_FLOW_ROIS
        for roi_name, roi_norm in OPTICAL_FLOW_ROIS.items():
            rx, ry, rw, rh = roi_norm
            x1 = int(rx * w);  y1 = int(ry * h)
            x2 = int((rx+rw)*w); y2 = int((ry+rh)*h)
            roi_res = flow.roi_results.get(roi_name)
            if roi_res and roi_res.active:
                cv2.rectangle(canvas, (x1,y1), (x2,y2), C_CYAN, 1)
                dx = int(roi_res.dx * 5)
                dy = int(roi_res.dy * 5)
                cx = (x1+x2)//2; cy = (y1+y2)//2
                cv2.arrowedLine(canvas, (cx, cy),
                                (cx+dx, cy+dy), C_CYAN, 2, tipLength=0.4)
            else:
                cv2.rectangle(canvas, (x1,y1), (x2,y2), C_DARK, 1)

    def _draw_graph_panel(self, canvas, result: FusionResult, w: int, h: int):
        """Right-side panel showing graph node activation weights."""
        gs = result.graph_state
        if gs is None:
            return
        panel_w = 220
        panel_x = w - panel_w - 5
        panel_y = 10

        # Semi-transparent dark background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (w-5, panel_y + len(gs.weights)*20 + 25),
                      C_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        cv2.putText(canvas, "SFG Activations",
                    (panel_x+5, panel_y+15), FONT, 0.42, C_WHITE, 1)

        sorted_nodes = sorted(gs.weights.items(), key=lambda x: -x[1])
        act_thr = self.cfg.thresholds.graph_activation_threshold

        for i, (node, weight) in enumerate(sorted_nodes):
            y = panel_y + 25 + i * 20
            bar_w = int(weight * (panel_w - 100))
            # Background bar
            cv2.rectangle(canvas, (panel_x+5, y),
                          (panel_x+5 + panel_w-105, y+12), (50,50,50), -1)
            # Fill bar
            col = C_GREEN if weight >= act_thr else C_GRAY
            cv2.rectangle(canvas, (panel_x+5, y),
                          (panel_x+5+bar_w, y+12), col, -1)
            # Label (short)
            label = node.split("(")[0][:12]
            cv2.putText(canvas, f"{label} {weight:.2f}",
                        (panel_x+5, y+10), FONT, 0.32, C_WHITE, 1)

    def _draw_semantic_banner(self, canvas, result: FusionResult, w: int, h: int):
        """Bottom banner with the final semantic output tokens."""
        banner_h = 50
        overlay  = canvas.copy()
        cv2.rectangle(overlay, (0, h-banner_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

        text = result.semantic_output
        col  = C_YELLOW if text != "NEUTRAL" else C_GRAY

        # Scale font to fit width
        scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, scale, thickness)
        x = max(10, (w - tw) // 2)
        cv2.putText(canvas, text, (x, h-14), FONT_BOLD, scale, col, thickness)

    def _draw_fps(self, canvas, result: FusionResult):
        import time
        now = time.perf_counter()
        if self._last_ts > 0:
            fps = 1.0 / max(now - self._last_ts, 1e-9)
            self._fps_buf.append(fps)
            if len(self._fps_buf) > 30:
                self._fps_buf.pop(0)
            avg_fps = float(np.mean(self._fps_buf))
        else:
            avg_fps = 0.0
        self._last_ts = now
        pt_ms = result.process_time_ms
        cv2.putText(canvas,
                    f"FPS:{avg_fps:4.1f}  Proc:{pt_ms:4.1f}ms",
                    (10, canvas.shape[0]-90), FONT, 0.42, C_WHITE, 1)
