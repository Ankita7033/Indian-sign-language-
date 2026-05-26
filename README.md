# ISL Non-Manual Feature Extraction System

**Real-time Indian Sign Language grammar analysis from webcam — zero training data required.**

> Converts face + body signals into structured ISL linguistic tokens using a novel Semantic Fusion Graph.

```
QUESTION(type=WH)   NEGATION(active)   EMPHASIS(strong)   TOPIC_SHIFT(true)
```

---

## What It Does

Most ISL systems recognize hand signs. This system reads the **grammatical layer** — the face, head, and shoulder signals that change the *meaning* of a sign:

| Signal | ISL Meaning |
|--------|-------------|
| Both eyebrows raised + forward gaze | WH-Question marker |
| Head shake left-right + furrowed brows | Negation |
| Shoulder shrug + furrowed brows | Doubt / Uncertainty |
| Head nod + wide eyes | Agreement / Emphasis |
| Head tilt + lateral gaze | Topic Shift |

Seven channels are extracted simultaneously and fused into a single linguistic interpretation via the **Semantic Fusion Graph** — a novel weighted directed graph where nodes are linguistic concepts and edges encode ISL grammatical relationships.

---

## System Architecture

```
Webcam/Video
    │
    ▼
FaceLandmarkExtractor (MediaPipe FaceMesh 478pts + Pose 33pts)
    │
    ├──► HeadPoseEstimator     solvePnP → pitch / yaw / roll
    ├──► EyebrowTracker        brow height, furrow, asymmetry
    ├──► EyeTracker            EAR blink, iris gaze vector
    ├──► LipContourExtractor   MAR, spread, hull area, protrusion
    ├──► ShoulderTracker       raise, lean, shrug (30-frame calibration)
    ├──► OpticalFlowTracker    Farnebäck dense flow, 6 facial ROIs
    └──► TemporalSmoother      EMA filters + gesture segmenter
              │
              ▼
    ┌─────────────────────────────────────────┐
    │   Semantic Fusion Graph  (Novel)        │
    │   G=(V,E,W)  16 nodes  10 edges        │
    │   decay → evidence → update → propagate│
    └─────────────────────────────────────────┘
              │
    ┌─────────┴──────────┬──────────────────┐
    ▼                    ▼                  ▼
TextGenerator      ExplainabilityEngine  Visualizer
    │                    │
    ▼                    ▼
"QUESTION(type=WH)"   "Detected because: bilateral brow raise + forward gaze"
```

---

## Installation

```bash
pip install mediapipe==0.10.13 opencv-python numpy scipy networkx scikit-learn pandas matplotlib tqdm
```

> ⚠️ Use exactly `mediapipe==0.10.13` — newer versions removed the required API.

---

## Usage

```bash
# Real-time webcam
python main.py --webcam

# With feature-level explanation of why each token fired
python main.py --webcam --explain

# Process a video file
python main.py --video signing.mp4

# Single image
python main.py --image frame.jpg

# Save annotated output video
python main.py --video input.mp4 --save-output output.mp4

# With evaluation metrics (latency, SAS score)
python main.py --webcam --eval

# Generate architecture diagram
python -m utils.architecture_diagram

# Generate Semantic Fusion Graph visualization
python -m semantic_graph.graph_visualizer

# Run all tests
python tests/test_pipeline.py
```

**Keyboard controls during webcam:** `q`/`ESC` = quit · `r` = reset graph

---

## Example Output

### Standard mode
```
[10:21:07] Frame 00036 | 🔴 QUESTION(type=WH) EMPHASIS(strong)
         [WH-Question: raised brows + forward gaze]
         [Strong Emphasis: nod + wide eyes + raised shoulders]
```

### Explain mode (`--explain`)
```
────────────────────────────────────────────────────────
  FRAME 00036 — Semantic Explanation
────────────────────────────────────────────────────────

  ► QUESTION(type=WH)  [weight=0.814]
    Detected because: bilateral brow raise + forward gaze + rapid brow movement
    ✦ bilateral brow raise    ██████████  0.91
    ✦ forward gaze            ████████░░  0.78
    ◈ rapid brow movement     ██████░░░░  0.61
    ◦ open mouth              ████░░░░░░  0.42

  ► EMPHASIS(strong)  [weight=0.721]
    Detected because: head nod + wide-open eyes + bilateral shoulder raise
    ✦ head nod                ████████░░  0.83
    ✦ wide-open eyes          ███████░░░  0.71
    ◈ bilateral shoulder raise████░░░░░░  0.44
────────────────────────────────────────────────────────
```

---

## Novel Contributions

1. **Semantic Fusion Graph (SFG)** — first use of a weighted directed linguistic concept graph for ISL NMS fusion. No training data required.
2. **Multi-channel holistic output** — 7 simultaneous channels → 1 joint linguistic interpretation
3. **Semantic Alignment Score (SAS)** — novel evaluation metric based on token-set Jaccard similarity weighted by annotator confidence
4. **Explainability Engine** — per-token feature-level reasoning with visual bar indicators

---

## Project Structure

```
isl_nmf_system/
├── main.py                          Entry point
├── config/config.py                 All thresholds & landmark indices
├── feature_extractors/              7 extraction modules + temporal smoother
├── semantic_graph/                  Semantic Fusion Graph + visualizer
├── fusion_engine/                   Orchestrator + TextGenerator + Explainability
├── evaluation/                      Metrics + Ablation study
├── datasets/                        ISLRTC-compatible annotation schema
├── visualizer/                      OpenCV overlay renderer
├── utils/                           Math helpers, logger, architecture diagram
└── tests/test_pipeline.py           10 unit + integration tests
```

---

## Evaluation Protocol

No pre-existing ISL NMS dataset is used. Evaluation follows an annotation protocol compatible with ISLRTC guidelines:

- Annotators label frame segments with active linguistic tokens + confidence score
- `EvaluationModule` computes per-class P/R/F1 and SAS
- `AblationStudy` quantifies each channel's contribution by zeroing it out

Run: `python main.py --webcam --eval` → generates `evaluation/results/report.txt`

---

## Research Paper

An IEEE-format research paper derived from this implementation is included as `ISL_NMF_Research_Paper.docx`, covering Abstract, Introduction, Literature Review, Research Gap, Methodology, System Architecture, Mathematical Modeling, Experimental Setup, Results, Ablation Study, Applications, Limitations, Future Work, and Conclusion.
