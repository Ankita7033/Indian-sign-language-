"""
main.py — ISL NMF System v5.0 — COMPLETE BUILD
================================================
19 features integrated. Production-grade ISL interpreter.

Commands:
  python main.py --webcam
  python main.py --webcam --explain
  python main.py --webcam --hands
  python main.py --webcam --calibrate-user
  python main.py --webcam --record --auto-annotate
  python main.py --webcam --edge-mode
  python main.py --webcam --multi-person
  python main.py --webcam --eval
  python main.py --video PATH --save-output out.mp4
"""

import sys, os, argparse, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import SystemConfig
from fusion_engine.fusion_engine              import FusionEngine
from fusion_engine.text_generator             import TextGenerator
from fusion_engine.explainability             import ExplainabilityEngine
from fusion_engine.confidence_scorer          import ConfidenceScorer
from fusion_engine.temporal_memory            import TemporalMemory
from fusion_engine.subtitle_generator         import SubtitleGenerator
from fusion_engine.isl_interpreter            import ISLInterpreter
from fusion_engine.grammar_engine             import GrammarEngine
from fusion_engine.interaction_state          import InteractionStateDetector, STATE_COLORS
from fusion_engine.grammar_validator          import GrammarValidator
from fusion_engine.semantic_priority_resolver import SemanticPriorityResolver
from fusion_engine.caption_streaming_api      import CaptionStreamingAPI
from fusion_engine.emotion_grammar_model      import EmotionGrammarModel
from fusion_engine.isl_grammar_rule_engine    import ISLGrammarRuleEngine
from fusion_engine.semantic_confidence_fusion import SemanticConfidenceFusionEngine
from feature_extractors.hand_gesture          import HandGestureDetector
from feature_extractors.multi_person_tracker  import MultiPersonTracker
from feature_extractors.eye_gaze_intent       import EyeGazeIntentPredictor
from visualizer.visualizer                    import Visualizer
from visualizer.timeline_visualizer           import TimelineVisualizer
from visualizer.decision_graph_viewer         import DecisionGraphViewer
from visualizer.research_dashboard            import ResearchDashboard
from utils.user_calibration                   import (load_profile, save_profile,
                                                       UserCalibrator,
                                                       apply_profile_to_config,
                                                       CALIBRATION_FRAMES)
from utils.edge_optimizer                     import EdgeOptimizer, EdgeConfig
from utils.adaptive_threshold_engine          import AdaptiveThresholdEngine
from utils.lighting_adaptation                import LightingAdaptationEngine
from datasets.dataset_recorder               import DatasetRecorder
from datasets.auto_annotation_engine         import AutoAnnotationEngine
from evaluation.evaluation_metrics_engine    import EvaluationMetricsEngine
from utils.logger import get_logger

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="ISL NMF System v5.0")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam",         action="store_true", default=True)
    src.add_argument("--video",          type=str, metavar="PATH")
    src.add_argument("--image",          type=str, metavar="PATH")
    p.add_argument("--camera-id",        type=int, default=0)
    p.add_argument("--no-viz",           action="store_true")
    p.add_argument("--eval",             action="store_true")
    p.add_argument("--explain",          action="store_true")
    p.add_argument("--hands",            action="store_true")
    p.add_argument("--multi-person",     action="store_true")
    p.add_argument("--calibrate-user",   action="store_true")
    p.add_argument("--record",           action="store_true")
    p.add_argument("--auto-annotate",    action="store_true")
    p.add_argument("--edge-mode",        action="store_true")
    p.add_argument("--no-lighting",      action="store_true",
                   help="Disable lighting adaptation")
    p.add_argument("--debug",            action="store_true")
    p.add_argument("--save-output",      type=str, metavar="PATH")
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps",    type=int, default=30)
    return p.parse_args()


def build_config(args):
    cfg = SystemConfig()
    cfg.debug_mode          = args.debug
    cfg.show_visualization  = not args.no_viz
    cfg.camera.device_id    = args.camera_id
    cfg.camera.frame_width  = args.width
    cfg.camera.frame_height = args.height
    cfg.camera.flip_horizontal = not bool(args.video)
    profile = load_profile()
    if profile.calibrated:
        apply_profile_to_config(profile, cfg)
        log.info("User profile applied.")
    return cfg


def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS,          args.fps)
    if not cap.isOpened():
        log.error("Cannot open capture."); sys.exit(1)
    return cap


def run_calibration(args, cfg):
    cap        = open_capture(args)
    calibrator = UserCalibrator()
    engine     = FusionEngine(cfg)
    print("\n" + "="*60)
    print("  USER CALIBRATION — sit neutral for 10 seconds")
    print("="*60)
    time.sleep(2)
    for cd in [3, 2, 1]:
        print(f"  {cd}..."); time.sleep(1)
    print("  GO!\n")
    for fi in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret: continue
        if cfg.camera.flip_horizontal:
            frame = cv2.flip(frame, 1)
        result = engine.process_frame(frame, fi)
        h = result.head_pose; e = result.eye; l = result.lip
        prog = calibrator.feed_frame(
            result.feature_vector,
            head_pitch=h.pitch_deg if h else 0.0,
            head_yaw=h.yaw_deg   if h else 0.0,
            head_roll=h.roll_deg  if h else 0.0,
            ear=e.mean_ear       if e else 0.30,
            mar=l.mar            if l else 0.08,
        )
        if fi % 30 == 0:
            bar = "█"*int(prog*20) + "░"*(20-int(prog*20))
            print(f"  [{bar}] {int(prog*100)}%")
        if cfg.show_visualization:
            cv2.putText(frame, f"CALIBRATING {int(prog*100)}%",
                        (40,60), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (50,220,50), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
    profile = calibrator.compute_profile()
    save_profile(profile)
    cap.release(); cv2.destroyAllWindows(); engine.close()
    print(f"\n✅ Calibration saved.")
    print(f"   Eyebrow threshold: {profile.eyebrow_raise_threshold:.4f}")
    print(f"   Nod threshold    : {profile.head_nod_threshold:.1f}°")


def run_pipeline(args, cfg):
    cap = open_capture(args)

    # ── Module initialization ────────────────────────────────────
    engine       = FusionEngine(cfg)
    text_gen     = TextGenerator()
    explainer    = ExplainabilityEngine()
    confidence   = ConfidenceScorer()
    conf_fusion  = SemanticConfidenceFusionEngine()
    memory       = TemporalMemory(min_frames=6, clear_frames=10)
    subtitles    = SubtitleGenerator()
    interpreter  = ISLInterpreter()
    grammar      = GrammarEngine()
    grammar_rules= ISLGrammarRuleEngine()
    interaction  = InteractionStateDetector()
    validator    = GrammarValidator()
    resolver     = SemanticPriorityResolver()
    emotion_model= EmotionGrammarModel()
    gaze_intent  = EyeGazeIntentPredictor()
    caption_api  = CaptionStreamingAPI(fps=args.fps)
    adaptive_thr = AdaptiveThresholdEngine(cfg)
    lighting     = LightingAdaptationEngine(enabled=not args.no_lighting)
    viz          = Visualizer(cfg)
    timeline     = TimelineVisualizer()
    dec_graph    = DecisionGraphViewer()
    dashboard    = ResearchDashboard()
    evaluator    = EvaluationMetricsEngine() if args.eval else None
    hand_det     = HandGestureDetector() if args.hands else None
    multi_track  = MultiPersonTracker() if args.multi_person else None
    recorder     = DatasetRecorder() if args.record else None
    auto_annot   = AutoAnnotationEngine(fps=args.fps) if args.auto_annotate else None
    edge         = EdgeOptimizer(EdgeConfig(
                       enabled=args.edge_mode,
                       frame_skip=2, scale_factor=0.65))

    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_output, fourcc,
                                 args.fps, (args.width, args.height))

    frame_idx = 0
    fps_times = []
    # Defaults for before first processing
    confirmed    = ["NEUTRAL"]
    conf_scores  = {}
    fused_conf   = {}
    subtitle     = type('S', (), {'text': ''})()
    sentence     = type('S', (), {'english': ''})()
    int_state    = type('S', (), {'state': 'NEUTRAL', 'description': ''})()
    validation   = type('S', (), {'is_valid': True, 'warnings': [], 'summary': '✓'})()
    grammar_parse= type('S', (), {'active_structure': '', 'gloss': '', 'violations': []})()
    emotion      = type('S', (), {'dominant': 'NEUTRAL', 'confidence': 0.0})()
    gaze_result  = type('S', (), {'intent': 'NEUTRAL', 'description': ''})()
    hand_gesture = "none"
    light_stats  = type('S', (), {'condition': 'normal', 'correction_applied': 'none'})()
    result       = None

    print(f"\n{'='*64}")
    print("  ISL NMF System v5.0 — Full Feature Build")
    print(f"  Lighting: {'OFF' if args.no_lighting else 'ON'}  "
          f"Hands: {args.hands}  Edge: {args.edge_mode}")
    print(f"  Record: {args.record}  AutoAnnotate: {args.auto_annotate}")
    print("  Keys: q=quit  r=reset  R=record toggle  S=save recording")
    print(f"{'='*64}\n")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.video: break
                continue
            if cfg.camera.flip_horizontal and not args.video:
                frame = cv2.flip(frame, 1)

            ts = time.strftime("%H:%M:%S")

            # ── Lighting adaptation ──────────────────────────────
            proc_frame, light_stats = lighting.process(frame)

            # ── Edge mode ────────────────────────────────────────
            proc_frame = edge.preprocess(proc_frame)

            if edge.should_process(frame_idx):
                # ── Core NMF pipeline ────────────────────────────
                result      = engine.process_frame(proc_frame, frame_idx)
                gs          = result.graph_state
                raw_tok     = gs.token_sequence if gs else ["NEUTRAL"]
                fv          = result.feature_vector

                # ── Hand gesture ─────────────────────────────────
                hand_feat    = hand_det.process(proc_frame) if hand_det else None
                hand_gesture = hand_feat.combined_gesture if hand_feat else "none"

                # ── Multi-person ─────────────────────────────────
                persons = multi_track.update(proc_frame, frame_idx) \
                          if multi_track else []

                # ── Gaze intent ──────────────────────────────────
                if result.eye:
                    gaze_result = gaze_intent.update(
                        result.eye.gaze_x, result.eye.gaze_y, result.eye.mean_ear)

                # ── Emotion detection ────────────────────────────
                emotion = emotion_model.detect_emotion(fv)

                # ── Temporal memory ──────────────────────────────
                confirmed = memory.update(raw_tok, gs.weights if gs else {})

                # ── Priority resolver ────────────────────────────
                conf_pre = confidence.score(confirmed, fv, gs.weights if gs else {})
                resolved = resolver.resolve(confirmed, conf_pre, gs.weights if gs else {})
                confirmed = resolved.tokens

                # ── Semantic confidence fusion ───────────────────
                fused_conf = conf_fusion.fuse(confirmed, fv, gs.weights if gs else {})

                # ── Grammar rule engine ──────────────────────────
                grammar_parse = grammar_rules.parse(confirmed, fv)

                # ── Grammar validation ───────────────────────────
                validation = validator.validate(confirmed, fv, fused_conf)

                # ── Interaction state ────────────────────────────
                int_state = interaction.detect(confirmed, fv)

                # ── Grammar generation ───────────────────────────
                base_sentence = grammar.generate(confirmed, hand_gesture)
                enriched_en   = emotion_model.enrich(
                    confirmed, emotion, base_sentence.english)

                # ── Subtitle + Caption API ───────────────────────
                subtitle = subtitles.generate(confirmed, ts)
                mean_conf_pct = int(np.mean([v.fused_pct for v in fused_conf.values()])) \
                                if fused_conf else 50
                caption_api.push(frame_idx, confirmed,
                                 enriched_en, subtitle.text,
                                 mean_conf_pct, args.fps)

                # ── Adaptive threshold update ────────────────────
                is_neutral = confirmed == ["NEUTRAL"] or not confirmed
                adaptive_thr.feed(fv, is_neutral=is_neutral)
                if frame_idx % 90 == 0:
                    adaptive_thr.apply_to_config()

                # ── Auto-annotation ──────────────────────────────
                if auto_annot:
                    auto_annot.feed(frame_idx, confirmed, mean_conf_pct / 100)

                # ── Dataset recorder ─────────────────────────────
                if recorder:
                    recorder.feed_frame(
                        frame, frame_idx, fv, confirmed,
                        {t: v.fused_pct for t, v in fused_conf.items()},
                        result.landmark_result)

                # ── Timeline update ──────────────────────────────
                timeline.update(confirmed)

                # ── Dashboard update ─────────────────────────────
                text_out = text_gen.generate(confirmed, frame_idx)
                dashboard.update(
                    result.process_time_ms,
                    new_tokens  = text_out.is_new and confirmed != ["NEUTRAL"],
                    new_caption = bool(subtitle.text),
                    adapt_status = adaptive_thr.get_status().split("|")[0].strip()
                )

                # ── Evaluation logging ───────────────────────────
                if evaluator:
                    evaluator.log(frame_idx, confirmed,
                                  result.process_time_ms)

                edge.record_timing(result.process_time_ms)

                # ── Terminal output ──────────────────────────────
                if text_out.is_new:
                    output = " ".join(confirmed)
                    if output == "NEUTRAL":
                        if frame_idx % 150 == 0:
                            print(f"[{ts}] ● NEUTRAL  "
                                  f"[{int_state.state}] "
                                  f"gaze:{gaze_result.intent}")
                    else:
                        print(f"\n[{ts}] Frame {frame_idx:05d}")
                        print(f"  🔴 Tokens    : {output}")
                        conf_str = "  ".join(
                            f"{t.split('(')[0]}={v.fused_pct}%({v.grade})"
                            for t, v in fused_conf.items())
                        print(f"  📊 Confidence: {conf_str}")
                        print(f"  💬 Subtitle  : {subtitle.text}")
                        print(f"  📝 Sentence  : {enriched_en}")
                        print(f"  🎭 Mode      : {int_state.state}")
                        print(f"  👁 Gaze      : {gaze_result.intent} — {gaze_result.description}")
                        print(f"  😊 Emotion   : {emotion.dominant} ({emotion.confidence:.0%})")
                        print(f"  📐 Grammar   : {grammar_parse.active_structure}")
                        print(f"  💡 Gloss     : {grammar_parse.gloss}")
                        if grammar_parse.violations:
                            for v in grammar_parse.violations:
                                print(f"  ❌ {v}")
                        if not validation.is_valid:
                            for w in validation.warnings:
                                sev = "❌" if w.severity == "ERROR" else "⚠"
                                print(f"  {sev} {w.message}")
                                print(f"      → {w.suggestion}")
                        if hand_gesture != "none":
                            print(f"  🤲 Hand      : {hand_gesture}")
                        if args.explain and result and result.graph_state:
                            rpt = explainer.explain(
                                result.graph_state.active_nodes,
                                fv, result.graph_state.weights, frame_idx)
                            print(rpt.to_string())
                        if resolved.suppressed:
                            sup_short = [t.split("(")[0] for t in resolved.suppressed[:3]]
                            print(f"  🔇 Suppressed: {sup_short}")
                        if light_stats.condition != "normal":
                            print(f"  💡 Lighting  : {light_stats.condition} "
                                  f"→ {light_stats.correction_applied}")

            # ── Visualisation ─────────────────────────────────────
            if cfg.show_visualization and result:
                annotated = viz.draw(frame, result)
                h, w = annotated.shape[:2]

                # Research dashboard (top-left)
                dashboard.render(annotated, result.feature_vector,
                                 confirmed, x=5, y=5)

                # Timeline (bottom-left)
                timeline.render(annotated, 5, h-265, width=430, height=255)

                # Decision graph (bottom-right)
                if result.graph_state and result.graph_state.active_nodes:
                    dec_graph.render(
                        annotated,
                        result.graph_state.active_nodes,
                        result.feature_vector,
                        result.graph_state.weights,
                        x=w-395, y=h-320, width=388, height=310)

                # Interaction state
                s_col = STATE_COLORS.get(int_state.state, (120,120,120))
                cv2.rectangle(annotated, (5, 180), (220, 196), (20,20,20), -1)
                cv2.putText(annotated,
                            f"Mode: {int_state.state} | {gaze_result.intent}",
                            (10, 193),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_col, 1)

                # Emotion badge
                cv2.putText(annotated,
                            f"Emotion: {emotion.dominant} {emotion.confidence:.0%}",
                            (10, 208),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (200, 200, 100), 1)

                # Grammar structure
                if grammar_parse.active_structure:
                    cv2.putText(annotated,
                                f"Struct: {grammar_parse.active_structure[:30]}",
                                (10, 222),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                                (150, 200, 150), 1)

                # Validation badge
                val_col = (50,200,50) if validation.is_valid else (50,50,220)
                cv2.putText(annotated, validation.summary,
                            (10, 236),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, val_col, 1)

                # Fused confidence bars
                cy = 250
                for tok, fc in list(fused_conf.items())[:4]:
                    bw = int(fc.fused_score * 110)
                    cv2.rectangle(annotated, (10, cy), (120, cy+12), (35,35,35), -1)
                    col = (50,220,50) if fc.fused_pct>=75 else \
                          (50,200,220) if fc.fused_pct>=50 else (80,80,220)
                    cv2.rectangle(annotated, (10, cy), (10+bw, cy+12), col, -1)
                    cv2.putText(annotated,
                                f"{tok.split('(')[0][:8]}:{fc.fused_pct}%{fc.grade}",
                                (10, cy+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                (255,255,255), 1)
                    cy += 16

                # Sentence banner (bottom)
                display_sentence = enriched_en if enriched_en and enriched_en != "..." \
                                   else subtitle.text
                if display_sentence:
                    ov = annotated.copy()
                    cv2.rectangle(ov, (0, h-50), (w, h-1), (0,40,10), -1)
                    cv2.addWeighted(ov, 0.8, annotated, 0.2, 0, annotated)
                    cv2.putText(annotated, f"  {display_sentence}",
                                (10, h-20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.60,
                                (180, 255, 180), 1)

                # Lighting status
                if light_stats.condition != "normal":
                    cv2.putText(annotated,
                                f"Light: {light_stats.condition} "
                                f"[{light_stats.correction_applied}]",
                                (10, h-72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                (200,180,50), 1)

                # Recorder status
                if recorder:
                    rec_col = (50,50,220) if recorder.is_recording else (120,120,120)
                    cv2.putText(annotated, recorder.get_status_overlay(),
                                (10, h-88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, rec_col, 1)

                # Auto-annotation status
                if auto_annot:
                    cv2.putText(annotated, auto_annot.get_stats(),
                                (10, h-104),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                                (150, 200, 150), 1)

                # Multi-person
                if multi_track and persons:
                    multi_track.draw_tracks(annotated, persons)

                if writer: writer.write(annotated)
                cv2.imshow("ISL NMF System v5.0", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('r') and not recorder:
                    engine.sfg.reset(); text_gen.reset()
                    memory.reset(); subtitles.reset(); grammar.reset()
                elif recorder:
                    msg = recorder.handle_key(key)
                    if msg: print(f"  📹 {msg}")

            fps_times.append(time.perf_counter() - t0)
            if len(fps_times) > 30: fps_times.pop(0)
            frame_idx += 1

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        cap.release()
        if writer: writer.release()
        if hand_det: hand_det.close()
        cv2.destroyAllWindows()
        engine.close()

        # Export captions
        if caption_api.total_captions > 0:
            caption_api.export_srt()
            caption_api.export_vtt()
            caption_api.export_json()

        # Save auto-annotations
        if auto_annot:
            auto_annot.save()

        # Evaluation report
        if evaluator:
            report = evaluator.compute()
            print("\n" + report.to_paper_table())
            evaluator.save()

        if fps_times:
            mean_fps = 1.0 / max(float(np.mean(fps_times)), 1e-9)
            print(f"\n  {frame_idx} frames | Mean FPS: {mean_fps:.1f}")


def main():
    args = parse_args()
    if args.video or args.image:
        args.webcam = False
    cfg = build_config(args)
    log.info("ISL NMF System v5.0 starting.")
    if args.calibrate_user:
        run_calibration(args, cfg)
    else:
        run_pipeline(args, cfg)


if __name__ == "__main__":
    main()
