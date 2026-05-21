# ISL NMF System — Installation & Demo Guide

## Prerequisites
- Python 3.9 – 3.11
- Webcam (for real-time mode)
- Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2 supported)

## Environment Setup

```bash
# 1. Clone / unzip the project
cd isl_nmf_system

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# 4. (Optional) Install dlib with face detection
# Linux:
apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib==19.24.2

# 5. Run tests (no webcam required)
python tests/test_pipeline.py
```

## Demo Execution

### Real-time Webcam Mode
```bash
python main.py --webcam
```

### Process a Video File
```bash
python main.py --video path/to/sign_video.mp4
```

### Single Image
```bash
python main.py --image path/to/frame.jpg
```

### Headless (no display window)
```bash
python main.py --webcam --no-viz
```

### With Evaluation + Latency Logging
```bash
python main.py --webcam --eval
```

### Save Annotated Output Video
```bash
python main.py --video input.mp4 --save-output annotated_output.mp4
```

### Visualise Semantic Fusion Graph Structure
```bash
python -m semantic_graph.graph_visualizer
# Produces: semantic_graph_structure.png
```

### Generate Sample Annotation File
```bash
python -m datasets.annotation_schema
# Produces: datasets/sample_annotation.json
```

## Keyboard Controls (webcam mode)
| Key | Action |
|-----|--------|
| q / ESC | Quit |
| r | Reset semantic graph weights |

## Semantic Output Format
The system outputs linguistic tokens in this format:
```
QUESTION(type=WH) NEGATION(active) EMPHASIS(strong) TOPIC_SHIFT(true)
```

NOT individual feature states like:
```
eyebrow_up=True  ← This is NOT the system output
```

## System Architecture Overview
```
Webcam/Video
    │
    ▼
FaceLandmarkExtractor (MediaPipe FaceMesh + Pose)
    │
    ├──► HeadPoseEstimator    (solvePnP, Euler angles)
    ├──► EyebrowTracker       (height, furrow, asymmetry)
    ├──► EyeTracker           (EAR, iris gaze)
    ├──► LipContourExtractor  (MAR, spread, protrusion)
    ├──► ShoulderTracker      (raise, lean, shrug)
    └──► OpticalFlowTracker   (Farnebäck, ROI analysis)
              │
              ▼
         TemporalSmoother (EMA + gesture segmentation)
              │
              ▼
    SemanticFusionGraph  ← NOVEL CONTRIBUTION
    (weighted digraph, evidence accumulation,
     one-hop propagation, hysteresis activation)
              │
              ▼
         TextGenerator
              │
              ▼
    "QUESTION(type=WH) NEGATION(active)"
```
