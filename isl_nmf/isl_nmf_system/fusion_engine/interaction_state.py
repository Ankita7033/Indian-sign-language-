"""
fusion_engine/interaction_state.py
=====================================
Speaker Mode vs Listener Mode Detection — Feature #2

Detects the signer's current communicative role and state:
  QUESTIONING   — asking something (brows up, forward lean)
  ASSERTING     — making a statement (steady gaze, neutral brows)
  EMPHASIZING   — stressing a point (wide eyes, strong nod)
  THINKING      — processing (gaze lateral/up, slight head tilt)
  LISTENING     — receiving (still face, sustained eye contact)
  NARRATING     — telling a story (topic shifts, smooth motion)

Uses: eye gaze direction, head orientation, facial tension,
      temporal consistency of non-manual markers.

Real-world use: interactive AI assistants, smart classrooms,
                meeting accessibility tools.
"""

from dataclasses import dataclass
from typing import Dict, List
from collections import deque
from config.config import LinguisticTokens

T = LinguisticTokens

# Interaction state definitions
class InteractionStates:
    QUESTIONING  = "QUESTIONING"
    ASSERTING    = "ASSERTING"
    EMPHASIZING  = "EMPHASIZING"
    THINKING     = "THINKING"
    LISTENING    = "LISTENING"
    NARRATING    = "NARRATING"
    NEUTRAL_STATE = "NEUTRAL"


# State icons for display
STATE_ICONS = {
    InteractionStates.QUESTIONING:  "❓",
    InteractionStates.ASSERTING:    "💬",
    InteractionStates.EMPHASIZING:  "❗",
    InteractionStates.THINKING:     "💭",
    InteractionStates.LISTENING:    "👂",
    InteractionStates.NARRATING:    "📖",
    InteractionStates.NEUTRAL_STATE: "⬜",
}

STATE_COLORS = {
    InteractionStates.QUESTIONING:  (50, 220, 220),
    InteractionStates.ASSERTING:    (50, 200, 50),
    InteractionStates.EMPHASIZING:  (50, 50, 220),
    InteractionStates.THINKING:     (200, 180, 50),
    InteractionStates.LISTENING:    (180, 50, 200),
    InteractionStates.NARRATING:    (220, 130, 50),
    InteractionStates.NEUTRAL_STATE:(120, 120, 120),
}


@dataclass
class InteractionResult:
    state: str
    confidence: float
    description: str
    speaker_active: bool    # True = speaking/signing
    listener_active: bool   # True = listening/receiving


# State scoring rules — each state gets a score from features
def _score_states(tokens: List[str],
                  fv: Dict[str, float]) -> Dict[str, float]:
    scores: Dict[str, float] = {s: 0.0 for s in [
        InteractionStates.QUESTIONING,
        InteractionStates.ASSERTING,
        InteractionStates.EMPHASIZING,
        InteractionStates.THINKING,
        InteractionStates.LISTENING,
        InteractionStates.NARRATING,
        InteractionStates.NEUTRAL_STATE,
    ]}

    tok = set(tokens)

    # QUESTIONING: WH or YN question active
    if T.QUESTION_WH in tok:  scores[InteractionStates.QUESTIONING] += 0.80
    if T.QUESTION_YN in tok:  scores[InteractionStates.QUESTIONING] += 0.65
    scores[InteractionStates.QUESTIONING] += fv.get("both_raised", 0) * 0.30
    scores[InteractionStates.QUESTIONING] += fv.get("gaze_forward", 0) * 0.15

    # ASSERTING: neutral tokens + stable face
    if T.ASSERTION in tok:    scores[InteractionStates.ASSERTING]   += 0.70
    scores[InteractionStates.ASSERTING] += fv.get("face_stable", 0) * 0.40
    scores[InteractionStates.ASSERTING] += fv.get("gaze_forward", 0) * 0.25
    if T.NEGATION in tok:     scores[InteractionStates.ASSERTING]   += 0.25

    # EMPHASIZING: strong emphasis tokens
    if T.EMPHASIS_STRONG in tok: scores[InteractionStates.EMPHASIZING] += 0.85
    if T.EXCLAMATION in tok:     scores[InteractionStates.EMPHASIZING] += 0.60
    if T.FOCUS in tok:           scores[InteractionStates.EMPHASIZING] += 0.40
    scores[InteractionStates.EMPHASIZING] += fv.get("wide_eye", 0) * 0.30
    scores[InteractionStates.EMPHASIZING] += fv.get("head_nod", 0) * 0.25

    # THINKING: lateral gaze + head tilt + uncertainty
    if T.DOUBT in tok:        scores[InteractionStates.THINKING] += 0.55
    if T.UNCERTAINTY in tok:  scores[InteractionStates.THINKING] += 0.55
    if T.CONDITIONAL in tok:  scores[InteractionStates.THINKING] += 0.40
    scores[InteractionStates.THINKING] += fv.get("gaze_lateral", 0) * 0.45
    scores[InteractionStates.THINKING] += fv.get("gaze_up", 0)      * 0.35
    scores[InteractionStates.THINKING] += fv.get("head_tilt", 0)    * 0.25

    # LISTENING: stable face, no active tokens, sustained eye contact
    if not tok or tok == {T.NEUTRAL}:
        scores[InteractionStates.LISTENING] += 0.60
    scores[InteractionStates.LISTENING] += fv.get("face_stable", 0) * 0.50
    scores[InteractionStates.LISTENING] -= fv.get("brow_velocity", 0) * 0.30

    # NARRATING: topic shifts + topic markers over time
    if T.TOPIC_SHIFT in tok:   scores[InteractionStates.NARRATING] += 0.55
    if T.TOPIC_MARKER in tok:  scores[InteractionStates.NARRATING] += 0.45
    if T.AGREEMENT in tok:     scores[InteractionStates.NARRATING] += 0.20

    # NEUTRAL: fallback
    scores[InteractionStates.NEUTRAL_STATE] = max(
        0.0, 0.5 - max(scores.values()) * 0.5
    )

    return scores


STATE_DESCRIPTIONS = {
    InteractionStates.QUESTIONING:  "Signer is asking a question",
    InteractionStates.ASSERTING:    "Signer is making a statement",
    InteractionStates.EMPHASIZING:  "Signer is emphasizing a point",
    InteractionStates.THINKING:     "Signer is thinking / considering",
    InteractionStates.LISTENING:    "Signer is listening / attentive",
    InteractionStates.NARRATING:    "Signer is narrating / explaining",
    InteractionStates.NEUTRAL_STATE:"Signer is in neutral state",
}


class InteractionStateDetector:
    """
    Detects communicative role of the signer per frame.
    Uses temporal smoothing to avoid rapid state flipping.
    """

    HISTORY = 10

    def __init__(self):
        self._score_history: Dict[str, deque] = {
            s: deque(maxlen=self.HISTORY)
            for s in STATE_DESCRIPTIONS
        }

    def detect(self,
               confirmed_tokens: List[str],
               feature_vector: Dict[str, float]) -> InteractionResult:

        raw_scores = _score_states(confirmed_tokens, feature_vector)

        # Temporal smoothing
        for state, score in raw_scores.items():
            self._score_history[state].append(score)

        smoothed = {
            s: sum(h) / len(h) if h else 0.0
            for s, h in self._score_history.items()
        }

        # Pick best state
        best_state = max(smoothed, key=smoothed.get)
        best_score = smoothed[best_state]

        speaker_active = best_state not in (
            InteractionStates.LISTENING,
            InteractionStates.NEUTRAL_STATE
        )

        return InteractionResult(
            state           = best_state,
            confidence      = min(best_score, 1.0),
            description     = STATE_DESCRIPTIONS[best_state],
            speaker_active  = speaker_active,
            listener_active = not speaker_active,
        )
