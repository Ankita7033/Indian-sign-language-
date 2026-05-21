"""
server.py — ISL NMF System Web Dashboard Backend
Flask server that bridges the REAL ISL pipeline with the web frontend.
Provides REST API + Server-Sent Events (SSE) streaming.

Usage:
    pip install flask flask-cors
    python server.py                   # start with webcam (real pipeline)
    python server.py --demo            # start with simulated data (no webcam needed)
    python server.py --camera-id 1     # use a specific camera
"""

import sys
import os
import json
import time
import math
import random
import argparse
import threading
import numpy as np

# Add project root to path so all modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Shared latest annotated frame (MJPEG stream) ────────────────────────────
_latest_frame_lock = threading.Lock()
_latest_frame_jpg  = None   # bytes of the latest JPEG-encoded annotated frame


def _store_frame(frame_bgr):
    """Encode a BGR frame as JPEG and store it for the MJPEG feed."""
    global _latest_frame_jpg
    import cv2
    ret, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        with _latest_frame_lock:
            _latest_frame_jpg = buf.tobytes()


# ── Shared live state dict (updated by pipeline thread) ──────────────────────
_state = {
    "frame_idx": 0,
    "tokens": ["NEUTRAL"],
    "sentence": "Waiting for signer...",
    "subtitle": "",
    "confidence": {},
    "emotion": "NEUTRAL",
    "emotion_conf": 0.0,
    "gaze_intent": "NEUTRAL",
    "gaze_x": 0.0,
    "gaze_y": 0.0,
    "mode": "NEUTRAL",
    "grammar_structure": "",
    "gloss": "",
    "hand_gesture": "none",
    "suppressed": [],
    "validation_ok": True,
    "validation_summary": "✓ Grammar well-formed",
    "lighting": "normal",
    "fps": 0.0,
    "latency_ms": 0.0,
    "session_frames": 0,
    "session_captions": 0,
    "channels": {
        "eyebrow": 0.0, "eye": 0.0, "head": 0.0,
        "lip": 0.0, "shoulder": 0.0, "flow": 0.0
    },
    "graph_weights": {},
    "timeline": [],
    "captions_log": [],
    "pipeline_mode": "demo",   # "live" or "demo"
}

_state_lock = threading.Lock()

# Token display colors
TOKEN_COLORS = {
    "QUESTION(type=WH)":  "#38bdf8",
    "QUESTION(type=YN)":  "#34d399",
    "NEGATION(active)":   "#f87171",
    "EMPHASIS(strong)":   "#fb923c",
    "EMPHASIS(mild)":     "#fbbf24",
    "AGREEMENT":          "#4ade80",
    "DISAGREEMENT":       "#f87171",
    "DOUBT":              "#c084fc",
    "UNCERTAINTY":        "#94a3b8",
    "SURPRISE":           "#f472b6",
    "EXCLAMATION":        "#fb923c",
    "TOPIC_SHIFT(true)":  "#a78bfa",
    "TOPIC_MARKER":       "#818cf8",
    "FOCUS":              "#22d3ee",
    "NEUTRAL":            "#475569",
}


# ═══════════════════════════════════════════════════════════════════════════════
# REAL PIPELINE — runs the actual ISL NMF system with webcam
# ═══════════════════════════════════════════════════════════════════════════════

def _run_real_pipeline(camera_id=0, width=1280, height=720, fps_target=30):
    """
    Runs the real ISL NMF pipeline on a background thread.
    Captures webcam frames, processes them through the full pipeline,
    and updates _state for the frontend.
    """
    import cv2
    from config.config import SystemConfig
    from fusion_engine.fusion_engine import FusionEngine
    from fusion_engine.text_generator import TextGenerator
    from fusion_engine.confidence_scorer import ConfidenceScorer
    from fusion_engine.semantic_confidence_fusion import SemanticConfidenceFusionEngine
    from fusion_engine.temporal_memory import TemporalMemory
    from fusion_engine.subtitle_generator import SubtitleGenerator
    from fusion_engine.grammar_engine import GrammarEngine
    from fusion_engine.isl_grammar_rule_engine import ISLGrammarRuleEngine
    from fusion_engine.interaction_state import InteractionStateDetector
    from fusion_engine.grammar_validator import GrammarValidator
    from fusion_engine.semantic_priority_resolver import SemanticPriorityResolver
    from fusion_engine.emotion_grammar_model import EmotionGrammarModel
    from feature_extractors.eye_gaze_intent import EyeGazeIntentPredictor
    from utils.lighting_adaptation import LightingAdaptationEngine
    from utils.adaptive_threshold_engine import AdaptiveThresholdEngine
    from utils.logger import get_logger

    log = get_logger("server.pipeline")
    log.info(f"Starting REAL pipeline — camera={camera_id} {width}x{height}")

    # Build config
    cfg = SystemConfig()
    cfg.camera.device_id = camera_id
    cfg.camera.frame_width = width
    cfg.camera.frame_height = height
    cfg.show_visualization = False   # no cv2.imshow on server

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)

    if not cap.isOpened():
        log.error("Cannot open webcam! Falling back to demo mode.")
        with _state_lock:
            _state["pipeline_mode"] = "demo"
        _simulate_pipeline()
        return

    # Initialize all modules
    engine       = FusionEngine(cfg)
    text_gen     = TextGenerator()
    confidence   = ConfidenceScorer()
    conf_fusion  = SemanticConfidenceFusionEngine()
    memory       = TemporalMemory(min_frames=6, clear_frames=10)
    subtitles    = SubtitleGenerator()
    grammar      = GrammarEngine()
    grammar_rules = ISLGrammarRuleEngine()
    interaction  = InteractionStateDetector()
    validator    = GrammarValidator()
    resolver     = SemanticPriorityResolver()
    emotion_model = EmotionGrammarModel()
    gaze_intent  = EyeGazeIntentPredictor()
    adaptive_thr = AdaptiveThresholdEngine(cfg)
    lighting     = LightingAdaptationEngine(enabled=True)

    with _state_lock:
        _state["pipeline_mode"] = "live"

    frame_idx = 0
    fps_times = []

    # Default state objects
    gaze_result = type('S', (), {'intent': 'NEUTRAL', 'description': ''})()
    emotion     = type('S', (), {'dominant': 'NEUTRAL', 'confidence': 0.0})()

    log.info("Pipeline modules initialised. Processing frames...")

    # Import cv2 for annotation drawing
    import cv2 as _cv2

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip horizontally for mirror effect
            if cfg.camera.flip_horizontal:
                frame = _cv2.flip(frame, 1)

            ts = time.strftime("%H:%M:%S")

            # ── Lighting adaptation ──────────────────────────────
            proc_frame, light_stats = lighting.process(frame)

            # ── Core NMF pipeline ────────────────────────────────
            result      = engine.process_frame(proc_frame, frame_idx)
            
            if result.landmark_result and not result.landmark_result.face_detected:
                with _state_lock:
                    _state.update({
                        "frame_idx":        frame_idx,
                        "tokens":           ["NO FACE"],
                        "sentence":         "No face detected.",
                        "subtitle":         "No face detected.",
                        "confidence":       {},
                        "emotion":          "NO FACE",
                        "emotion_conf":     0.0,
                        "gaze_intent":      "NONE",
                        "gaze_x":           0.0,
                        "gaze_y":           0.0,
                        "num_faces":        0,
                        "mode":             "offline",
                        "grammar_structure": "NONE",
                        "gloss":            "...",
                        "hand_gesture":     "none",
                        "validation_ok":    False,
                        "validation_summary": "Waiting for face...",
                    })
                time.sleep(0.01)
                continue

            gs          = result.graph_state
            raw_tok     = gs.token_sequence if gs else ["NEUTRAL"]
            fv          = result.feature_vector

            # ── Gaze intent ──────────────────────────────────────
            if result.eye:
                gaze_result = gaze_intent.update(
                    result.eye.gaze_x, result.eye.gaze_y, result.eye.mean_ear)

            # ── Emotion detection ────────────────────────────────
            emotion = emotion_model.detect_emotion(fv)

            # ── Temporal memory ──────────────────────────────────
            confirmed = memory.update(raw_tok, gs.weights if gs else {})

            # ── Priority resolver ────────────────────────────────
            conf_pre = confidence.score(confirmed, fv, gs.weights if gs else {})
            resolved = resolver.resolve(confirmed, conf_pre, gs.weights if gs else {})
            confirmed = resolved.tokens

            # ── Semantic confidence fusion ───────────────────────
            fused_conf = conf_fusion.fuse(confirmed, fv, gs.weights if gs else {})

            # ── Grammar rule engine ──────────────────────────────
            grammar_parse = grammar_rules.parse(confirmed, fv)

            # ── Grammar validation ───────────────────────────────
            validation = validator.validate(confirmed, fv, fused_conf)

            # ── Interaction state ────────────────────────────────
            int_state = interaction.detect(confirmed, fv)

            # ── Grammar generation ───────────────────────────────
            base_sentence = grammar.generate(confirmed, result.hand_gesture)
            enriched_en   = emotion_model.enrich(
                confirmed, emotion, base_sentence.english)

            # ── Subtitle ─────────────────────────────────────────
            subtitle = subtitles.generate(confirmed, ts)

            # ── Adaptive threshold update ────────────────────────
            is_neutral = confirmed == ["NEUTRAL"] or not confirmed
            adaptive_thr.feed(fv, is_neutral=is_neutral)
            if frame_idx % 90 == 0:
                adaptive_thr.apply_to_config()

            # ── Build channel values from feature vector ─────────
            channels = {
                "eyebrow":  float(min((fv.get("both_raised", 0) + fv.get("left_brow_raise", 0) + fv.get("right_brow_raise", 0)) / 2.0, 1.0)),
                "eye":      float(min((fv.get("mean_ear", 0) + fv.get("wide_eye", 0) + fv.get("gaze_forward", 0)) / 2.0, 1.0)),
                "head":     float(min((fv.get("head_nod", 0) + fv.get("is_shaking", 0) + fv.get("head_tilt", 0)) / 2.0, 1.0)),
                "lip":      float(min((fv.get("mouth_open", 0) + fv.get("lip_spread", 0) + fv.get("lip_protrusion", 0)) / 2.0, 1.0)),
                "shoulder": float(min((fv.get("shoulder_bilateral_raise", 0) + fv.get("is_shrugging", 0)) / 1.5, 1.0)),
                "flow":     float(min(fv.get("flow_active", 0), 1.0)),
            }

            # ── Build confidence dict for frontend ───────────────
            conf_dict = {}
            for tok, fc in fused_conf.items():
                conf_dict[tok] = fc.fused_pct

            # ── Gaze coordinates ─────────────────────────────────
            gx = result.eye.gaze_x if result.eye else 0.0
            gy = result.eye.gaze_y if result.eye else 0.0

            # ── Compute FPS ──────────────────────────────────────
            elapsed = time.perf_counter() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)
            current_fps = 1.0 / max(float(np.mean(fps_times)), 1e-9)
            latency_ms = result.process_time_ms

            # ── Update shared state ──────────────────────────────
            with _state_lock:
                _state.update({
                    "frame_idx":        frame_idx,
                    "tokens":           confirmed,
                    "sentence":         enriched_en if enriched_en and enriched_en != "..." else subtitle.text,
                    "subtitle":         subtitle.text,
                    "confidence":       conf_dict,
                    "emotion":          emotion.dominant,
                    "emotion_conf":     float(emotion.confidence),
                    "gaze_intent":      gaze_result.intent,
                    "gaze_x":           float(gx),
                    "gaze_y":           float(gy),
                    "num_faces":        result.landmark_result.num_faces if result.landmark_result else 1,
                    "mode":             int_state.state,
                    "grammar_structure": grammar_parse.active_structure,
                    "gloss":            grammar_parse.gloss,
                    "hand_gesture":     result.hand_gesture,
                    "suppressed":       [t.split("(")[0] for t in resolved.suppressed[:3]] if resolved.suppressed else [],
                    "validation_ok":    validation.is_valid,
                    "validation_summary": validation.summary,
                    "lighting":         light_stats.condition,
                    "fps":              round(current_fps, 1),
                    "latency_ms":       round(latency_ms, 1),
                    "session_frames":   frame_idx,
                    "session_captions": frame_idx // 90,
                    "channels":         channels,
                    "graph_weights":    dict(gs.weights) if gs else {},
                })

                # Add to captions log
                if frame_idx % 90 == 0 and confirmed != ["NEUTRAL"] and confirmed:
                    entry = {
                        "time": ts,
                        "tokens": confirmed,
                        "sentence": enriched_en if enriched_en else subtitle.text,
                        "confidence": max(conf_dict.values()) if conf_dict else 0,
                    }
                    _state["captions_log"] = ([entry] + _state["captions_log"])[:20]

            # ── Annotate frame for MJPEG stream ──────────────────
            disp = frame.copy()
            emo_txt  = f"{emotion.dominant} {int(emotion.confidence*100)}%"
            tok_txt  = " | ".join(confirmed[:3]) if confirmed else "NEUTRAL"
            sent_txt = (enriched_en or subtitle.text or "")[:60]

            # Semi-transparent top bar
            overlay = disp.copy()
            _cv2.rectangle(overlay, (0, 0), (disp.shape[1], 56), (8, 12, 20), -1)
            _cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)

            # Top-bar text
            _cv2.putText(disp, f"TOKENS: {tok_txt}",
                         (10, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, _cv2.LINE_AA)
            _cv2.putText(disp, f"EMO: {emo_txt}  GAZE: {gaze_result.intent}  MODE: {int_state.state}",
                         (10, 40), _cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, _cv2.LINE_AA)

            # Bottom sentence banner
            if sent_txt:
                bot_y = disp.shape[0]
                overlay2 = disp.copy()
                _cv2.rectangle(overlay2, (0, bot_y - 46), (disp.shape[1], bot_y), (0, 30, 10), -1)
                _cv2.addWeighted(overlay2, 0.8, disp, 0.2, 0, disp)
                _cv2.putText(disp, sent_txt,
                             (12, bot_y - 16), _cv2.FONT_HERSHEY_DUPLEX, 0.58, (180, 255, 180), 1, _cv2.LINE_AA)

            # Confidence bars (right side)
            cx = disp.shape[1] - 170
            cy2 = 70
            for tok, fc_pct in list(conf_dict.items())[:4]:
                bw = int(fc_pct / 100 * 150)
                col_b = (50, 220, 50) if fc_pct >= 75 else (50, 200, 220) if fc_pct >= 50 else (80, 80, 220)
                _cv2.rectangle(disp, (cx, cy2), (cx + 150, cy2 + 10), (30, 30, 30), -1)
                _cv2.rectangle(disp, (cx, cy2), (cx + bw, cy2 + 10), col_b, -1)
                _cv2.putText(disp, f"{tok.split('(')[0][:10]}:{fc_pct}%",
                             (cx, cy2 + 9), _cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
                cy2 += 14

            _store_frame(disp)

            frame_idx += 1

    except Exception as e:
        log.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        engine.close()
        log.info("Pipeline stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO PIPELINE — simulated data for when no webcam is available
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_pipeline():
    """Simulates live ISL pipeline output for demo mode."""
    scenarios = [
        {
            "tokens": ["QUESTION(type=WH)", "FOCUS"],
            "sentence": "Where are you going? / What are you doing?",
            "subtitle": "What / Where / Who?",
            "emotion": "CONFUSED", "emotion_conf": 0.72,
            "gaze_intent": "QUESTIONING",
            "mode": "QUESTIONING",
            "grammar_structure": "TOPIC [NMM: brow-raise] QUESTION-WORD",
            "gloss": "What/Where/Who/When/Why/How?",
            "hand_gesture": "point",
            "confidence": {"QUESTION(type=WH)": 91, "FOCUS": 74},
        },
        {
            "tokens": ["NEGATION(active)"],
            "sentence": "No. / That is not right.",
            "subtitle": "No / Not...",
            "emotion": "ANGRY", "emotion_conf": 0.65,
            "gaze_intent": "EMPHASIZING",
            "mode": "ASSERTING",
            "grammar_structure": "SUBJECT PREDICATE [NMM: head-shake]",
            "gloss": "NOT / No / Do not",
            "hand_gesture": "open_palm",
            "confidence": {"NEGATION(active)": 94},
        },
        {
            "tokens": ["AGREEMENT"],
            "sentence": "Yes, I agree.",
            "subtitle": "Yes / I agree.",
            "emotion": "HAPPY", "emotion_conf": 0.80,
            "gaze_intent": "AFFIRMING",
            "mode": "ASSERTING",
            "grammar_structure": "SIGN [NMM: head-nod-repeated]",
            "gloss": "Yes / I agree / That is correct",
            "hand_gesture": "thumb",
            "confidence": {"AGREEMENT": 88},
        },
        {
            "tokens": ["DOUBT", "UNCERTAINTY"],
            "sentence": "I'm not sure about that. Maybe.",
            "subtitle": "I'm not sure...",
            "emotion": "FEARFUL", "emotion_conf": 0.55,
            "gaze_intent": "THINKING",
            "mode": "THINKING",
            "grammar_structure": "SIGN [NMM: shrug, brow-furrow]",
            "gloss": "Maybe / I don't know / Possibly",
            "hand_gesture": "none",
            "confidence": {"DOUBT": 79, "UNCERTAINTY": 71},
        },
        {
            "tokens": ["EMPHASIS(strong)", "FOCUS"],
            "sentence": "This is very important! Pay attention!",
            "subtitle": "[Strong emphasis]",
            "emotion": "ANGRY", "emotion_conf": 0.60,
            "gaze_intent": "EMPHASIZING",
            "mode": "EMPHASIZING",
            "grammar_structure": "SIGN [NMM: nod, wide-eyes, shoulder-raise]",
            "gloss": "Very / Really / Important!",
            "hand_gesture": "fist",
            "confidence": {"EMPHASIS(strong)": 85, "FOCUS": 77},
        },
        {
            "tokens": ["NEUTRAL"],
            "sentence": "Waiting for gesture...",
            "subtitle": "",
            "emotion": "NEUTRAL", "emotion_conf": 0.90,
            "gaze_intent": "NEUTRAL",
            "mode": "NEUTRAL",
            "grammar_structure": "",
            "gloss": "[Neutral — no grammar rule active]",
            "hand_gesture": "none",
            "confidence": {},
        },
    ]

    frame = 0
    scenario_idx = 0
    hold_frames = 0
    hold_max = random.randint(60, 120)

    while True:
        time.sleep(1/30)
        frame += 1
        hold_frames += 1

        if hold_frames > hold_max:
            hold_frames = 0
            hold_max = random.randint(60, 150)
            scenario_idx = (scenario_idx + 1) % len(scenarios)

        sc = scenarios[scenario_idx]
        t  = frame / 30.0

        # Simulate live channel values
        channels = {
            "eyebrow":  abs(math.sin(t * 0.7)) * 0.9 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.1),
            "eye":      abs(math.sin(t * 1.1)) * 0.8 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.15),
            "head":     abs(math.sin(t * 0.5)) * 0.85 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.08),
            "lip":      abs(math.sin(t * 0.9)) * 0.7 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.05),
            "shoulder": abs(math.sin(t * 0.3)) * 0.6 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.05),
            "flow":     abs(math.sin(t * 1.3)) * 0.5 if sc["tokens"] != ["NEUTRAL"] else random.uniform(0, 0.1),
        }

        # Simulated gaze movement
        gx = math.sin(t * 0.4) * 0.3 if sc["gaze_intent"] != "NEUTRAL" else random.uniform(-0.05, 0.05)
        gy = math.cos(t * 0.3) * 0.2 if sc["gaze_intent"] != "NEUTRAL" else random.uniform(-0.05, 0.05)

        with _state_lock:
            _state.update({
                "frame_idx": frame,
                "tokens":    sc["tokens"],
                "sentence":  sc["sentence"],
                "subtitle":  sc["subtitle"],
                "confidence": sc["confidence"],
                "emotion":   sc["emotion"],
                "emotion_conf": sc["emotion_conf"],
                "gaze_intent": sc["gaze_intent"],
                "gaze_x":   gx,
                "gaze_y":   gy,
                "mode":      sc["mode"],
                "grammar_structure": sc["grammar_structure"],
                "gloss":     sc["gloss"],
                "hand_gesture": sc["hand_gesture"],
                "validation_ok": True,
                "validation_summary": "✓ Grammar well-formed",
                "lighting":  "normal",
                "fps":       round(28 + random.uniform(-3, 3), 1),
                "latency_ms": round(18 + random.uniform(-5, 8), 1),
                "session_frames": frame,
                "session_captions": frame // 90,
                "channels":  channels,
                "pipeline_mode": "demo",
            })

            # Add to captions log when sentence changes
            if frame % 90 == 0 and sc["tokens"] != ["NEUTRAL"]:
                entry = {
                    "time": time.strftime("%H:%M:%S"),
                    "tokens": sc["tokens"],
                    "sentence": sc["sentence"],
                    "confidence": max(sc["confidence"].values()) if sc["confidence"] else 0,
                }
                _state["captions_log"] = ([entry] + _state["captions_log"])[:20]


# ═══════════════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/api/video_feed")
def video_feed():
    """MJPEG stream of the annotated camera frame."""
    import cv2

    def generate():
        blank = None
        while True:
            time.sleep(1 / 20)   # 20 fps max
            with _latest_frame_lock:
                jpg = _latest_frame_jpg

            if jpg is None:
                # Generate a placeholder frame if no camera data yet
                if blank is None:
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    placeholder[:] = (8, 12, 20)
                    cv2.putText(placeholder, "Waiting for camera feed...",
                                (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (56, 189, 248), 2)
                    _, buf = cv2.imencode(".jpg", placeholder)
                    blank = buf.tobytes()
                jpg = blank

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory(".", "index.html")


@app.route("/api/state")
def get_state():
    with _state_lock:
        return jsonify(dict(_state))


@app.route("/api/token_colors")
def get_token_colors():
    return jsonify(TOKEN_COLORS)


@app.route("/api/pipeline_mode")
def get_pipeline_mode():
    """Returns whether we are running live or demo mode."""
    with _state_lock:
        return jsonify({"mode": _state["pipeline_mode"]})


@app.route("/api/stream")
def stream():
    """Server-Sent Events stream for real-time updates."""
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NpEncoder, self).default(obj)

    def event_stream():
        last_frame = -1
        while True:
            time.sleep(1/15)   # 15 updates/sec to browser
            with _state_lock:
                if _state["frame_idx"] != last_frame:
                    last_frame = _state["frame_idx"]
                    data = json.dumps({
                        "frame_idx":        _state["frame_idx"],
                        "tokens":           _state["tokens"],
                        "sentence":         _state["sentence"],
                        "subtitle":         _state["subtitle"],
                        "confidence":       _state["confidence"],
                        "emotion":          _state["emotion"],
                        "emotion_conf":     _state["emotion_conf"],
                        "gaze_intent":      _state["gaze_intent"],
                        "gaze_x":           _state.get("gaze_x", 0),
                        "gaze_y":           _state.get("gaze_y", 0),
                        "mode":             _state["mode"],
                        "channels":         _state["channels"],
                        "fps":              _state["fps"],
                        "latency_ms":       _state["latency_ms"],
                        "grammar_structure": _state["grammar_structure"],
                        "gloss":            _state["gloss"],
                        "hand_gesture":     _state["hand_gesture"],
                        "suppressed":       _state.get("suppressed", []),
                        "validation_ok":    _state.get("validation_ok", True),
                        "validation_summary": _state.get("validation_summary", "✓"),
                        "lighting":         _state.get("lighting", "normal"),
                        "session_frames":   _state["session_frames"],
                        "session_captions": _state["session_captions"],
                        "captions_log":     _state["captions_log"][:5],
                        "pipeline_mode":    _state["pipeline_mode"],
                    }, cls=NpEncoder)
                    yield f"data: {data}\n\n"
    return Response(event_stream(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/export/srt")
def export_srt():
    """Export current captions as SRT."""
    with _state_lock:
        log = list(reversed(_state["captions_log"]))
    lines = []
    for i, entry in enumerate(log):
        lines.append(str(i+1))
        start = f"00:00:{(i*3):02d},000"
        end   = f"00:00:{(i*3+2):02d},999"
        lines.append(f"{start} --> {end}")
        lines.append(entry["sentence"])
        lines.append("")
    srt = "\n".join(lines)
    return Response(srt, mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=isl_captions.srt"})


@app.route("/api/export/json")
def export_json():
    """Export session data as JSON."""
    with _state_lock:
        log = list(_state["captions_log"])
    return Response(
        json.dumps(log, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=isl_session.json"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_server_args():
    p = argparse.ArgumentParser(description="ISL NMF Dashboard Server")
    p.add_argument("--demo", action="store_true",
                   help="Run in demo mode with simulated data (no webcam)")
    p.add_argument("--camera-id", type=int, default=0,
                   help="Camera device ID (default: 0)")
    p.add_argument("--width", type=int, default=1280,
                   help="Camera frame width")
    p.add_argument("--height", type=int, default=720,
                   help="Camera frame height")
    p.add_argument("--port", type=int, default=5000,
                   help="Server port (default: 5000)")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Server host (default: 0.0.0.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_server_args()

    print("\n" + "="*64)
    print("  ISL NMF System — Web Dashboard Server")
    print("="*64)

    if args.demo:
        print("  Mode     : DEMO (simulated pipeline)")
        print(f"  Open     : http://localhost:{args.port}")
        print("="*64 + "\n")
        _state["pipeline_mode"] = "demo"
        threading.Thread(target=_simulate_pipeline, daemon=True).start()
    else:
        print("  Mode     : LIVE (real webcam pipeline)")
        print(f"  Camera   : {args.camera_id}")
        print(f"  Resolution: {args.width}x{args.height}")
        print(f"  Open     : http://localhost:{args.port}")
        print("="*64 + "\n")
        _state["pipeline_mode"] = "live"
        threading.Thread(
            target=_run_real_pipeline,
            args=(args.camera_id, args.width, args.height),
            daemon=True
        ).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
