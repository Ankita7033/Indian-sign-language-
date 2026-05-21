# ISL NMF System — Web Dashboard

## Start the Dashboard

```bash
# Install dependencies (once)
pip install flask flask-cors

# Start server
python server.py

# Open browser
http://localhost:5000
```

## What You See

The dashboard has 3 columns + a hero token bar:

**Hero Bar (top)** — Active grammar tokens as colored chips + English sentence

**Left Column**
- 6 channel activation bars (eyebrow, eye, head, lip, shoulder, flow)
- Emotion detection badge with icon
- Gaze compass (dot shows iris direction)
- Interaction mode + hand gesture

**Centre Column**
- ISL grammar structure (formal notation)
- Semantic confidence fusion bars (A/B/C/D/F grades)
- Grammar timeline (90-frame history per token)
- Session metrics (FPS, latency, frames, captions)

**Right Column**
- Live caption log with timestamps
- English subtitle
- System info (frame, lighting, validation)
- Export buttons (SRT, JSON, API)

## Connect to Real Pipeline

In `server.py`, replace the `_simulate_pipeline()` function
with imports from your ISL system:

```python
from fusion_engine.fusion_engine import FusionEngine
# Push real results to _state dict in your webcam loop
```

## Export Formats

- **SRT** — Standard subtitle file for video players
- **JSON** — Full session data with timestamps and confidence
- **API** — Raw JSON state at `/api/state`
- **SSE Stream** — Real-time at `/api/stream`
