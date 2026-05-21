"""
datasets/annotation_schema.py
================================
ISLRTC-compatible annotation schema and sample annotation loader.

This module defines the annotation format used for evaluation.
No pre-existing deep-learning dataset is used. Instead, annotations
are produced following ISLRTC video annotation guidelines adapted
for non-manual features.

Protocol
--------
An annotator watches a video and marks, per frame range:
  - Which linguistic tokens are active
  - Confidence of annotation (1 = certain, 0.5 = uncertain)

The schema is stored as JSON. A sample synthetic annotation
for testing the evaluation pipeline is included below.

JSON format
-----------
{
  "video_id": "sample_001",
  "fps": 30,
  "annotations": [
    {"frame_start": 0, "frame_end": 45,
     "tokens": ["QUESTION(type=WH)", "EMPHASIS(strong)"],
     "confidence": 0.9},
    ...
  ]
}
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config.config import LinguisticTokens
from evaluation.evaluation_metrics import FrameAnnotation
from utils.logger import get_logger

log = get_logger(__name__)
T = LinguisticTokens


@dataclass
class AnnotationSegment:
    frame_start: int
    frame_end:   int
    tokens: List[str]
    confidence: float = 1.0
    notes: str = ""


@dataclass
class VideoAnnotation:
    video_id: str
    fps: int
    annotator: str = "human"
    segments: List[AnnotationSegment] = field(default_factory=list)


def load_annotation_json(path: str) -> VideoAnnotation:
    """Load annotation JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = [
        AnnotationSegment(
            frame_start = s["frame_start"],
            frame_end   = s["frame_end"],
            tokens      = s["tokens"],
            confidence  = s.get("confidence", 1.0),
            notes       = s.get("notes", "")
        )
        for s in data.get("annotations", [])
    ]
    return VideoAnnotation(
        video_id  = data.get("video_id", "unknown"),
        fps       = data.get("fps", 30),
        annotator = data.get("annotator", "human"),
        segments  = segs
    )


def expand_to_frame_annotations(va: VideoAnnotation) -> List[FrameAnnotation]:
    """
    Expand segment-level annotations to per-frame FrameAnnotation list.
    Each frame in a segment gets the segment's token set.
    """
    frame_anns: List[FrameAnnotation] = []
    for seg in va.segments:
        for fi in range(seg.frame_start, seg.frame_end + 1):
            frame_anns.append(FrameAnnotation(
                frame_idx  = fi,
                tokens     = seg.tokens,
                confidence = seg.confidence
            ))
    return frame_anns


def create_sample_annotation(save_path: str = "datasets/sample_annotation.json"):
    """
    Creates a sample annotation JSON for testing.
    Represents ~10 seconds of ISL signing at 30 fps.
    Tokens chosen from ISLRTC-documented non-manual patterns.
    """
    annotation = {
        "video_id": "sample_isl_001",
        "fps": 30,
        "annotator": "protocol_v1",
        "annotations": [
            {
                "frame_start": 0,  "frame_end": 29,
                "tokens": [T.NEUTRAL],
                "confidence": 1.0,
                "notes": "Rest/neutral posture"
            },
            {
                "frame_start": 30, "frame_end": 74,
                "tokens": [T.QUESTION_WH, T.EMPHASIS_MILD],
                "confidence": 0.9,
                "notes": "WH-question with eyebrow raise"
            },
            {
                "frame_start": 75, "frame_end": 104,
                "tokens": [T.NEGATION],
                "confidence": 0.95,
                "notes": "Head shake + furrowed brows"
            },
            {
                "frame_start": 105, "frame_end": 149,
                "tokens": [T.EMPHASIS_STRONG, T.FOCUS],
                "confidence": 0.85,
                "notes": "Strong emphasis: nod + wide eyes"
            },
            {
                "frame_start": 150, "frame_end": 179,
                "tokens": [T.TOPIC_SHIFT],
                "confidence": 0.80,
                "notes": "Head tilt indicating topic shift"
            },
            {
                "frame_start": 180, "frame_end": 209,
                "tokens": [T.DOUBT, T.UNCERTAINTY],
                "confidence": 0.88,
                "notes": "Shoulder shrug + furrowed brows"
            },
            {
                "frame_start": 210, "frame_end": 239,
                "tokens": [T.QUESTION_YN, T.AGREEMENT],
                "confidence": 0.90,
                "notes": "YN-question: head nod pattern"
            },
            {
                "frame_start": 240, "frame_end": 270,
                "tokens": [T.SURPRISE, T.EXCLAMATION],
                "confidence": 0.82,
                "notes": "Wide eyes + open mouth + raised brows"
            },
        ]
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    log.info("Sample annotation saved to %s", save_path)
    return save_path


if __name__ == "__main__":
    path = create_sample_annotation()
    va   = load_annotation_json(path)
    anns = expand_to_frame_annotations(va)
    print(f"Loaded {len(anns)} frame annotations from {va.video_id}")
