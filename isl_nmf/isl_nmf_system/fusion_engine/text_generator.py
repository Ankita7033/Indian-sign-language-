"""
fusion_engine/text_generator.py
=================================
Converts the Semantic Fusion Graph token sequence into a human-readable
natural language annotation string for display and evaluation.

The TextGenerator maintains a token history buffer and applies:
  1. Token deduplication (suppress repeated identical outputs)
  2. Boundary detection (SENTENCE_BOUNDARY token)
  3. Narrative description generation for each active token
  4. Confidence-sorted output formatting

Output examples:
  Frame-level:  "QUESTION(type=WH) NEGATION(active)"
  Human label:  "[WH-Question] + [Active Negation]"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

from config.config import LinguisticTokens
from utils.logger import get_logger

log = get_logger(__name__)
T = LinguisticTokens


# Human-readable descriptions for each linguistic token
TOKEN_DESCRIPTIONS: Dict[str, str] = {
    T.QUESTION_WH:     "[WH-Question: raised brows + forward gaze]",
    T.QUESTION_YN:     "[YN-Question: head nod pattern detected]",
    T.NEGATION:        "[Active Negation: head-shake + furrowed brows]",
    T.ASSERTION:       "[Assertion: neutral forward posture]",
    T.EMPHASIS_STRONG: "[Strong Emphasis: nod + wide eyes + raised shoulders]",
    T.EMPHASIS_MILD:   "[Mild Emphasis: partial brow raise + slight nod]",
    T.TOPIC_SHIFT:     "[Topic Shift: lateral head tilt + gaze redirect]",
    T.CONDITIONAL:     "[Conditional: unilateral brow raise + tilt]",
    T.EXCLAMATION:     "[Exclamation: wide eyes + open mouth + elevated shoulders]",
    T.DOUBT:           "[Doubt/Uncertainty: shrug + furrowed brows]",
    T.SURPRISE:        "[Surprise: bilateral brow raise + wide eyes + open mouth]",
    T.AGREEMENT:       "[Agreement: repeated head nod]",
    T.DISAGREEMENT:    "[Disagreement: head shake]",
    T.UNCERTAINTY:     "[Uncertainty: shrug + lateral gaze]",
    T.CONFIRMATION:    "[Confirmation: downward nod + steady gaze]",
    T.TOPIC_MARKER:    "[Topic Marker: brow raise + head tilt]",
    T.FOCUS:           "[Focus: forward gaze + brow raise velocity]",
    T.BOUNDARY:        "--- SENTENCE BOUNDARY ---",
    T.NEUTRAL:         "[Neutral: no active non-manual markers]",
}


@dataclass
class TextOutput:
    raw_tokens: List[str] = field(default_factory=list)
    human_labels: List[str] = field(default_factory=list)
    formatted_string: str = ""
    is_new: bool = False           # True if output changed from previous frame
    frame_idx: int = 0


class TextGenerator:
    """
    Post-processes the Semantic Fusion Graph token sequence into
    labelled output strings with temporal deduplication.
    """

    def __init__(self, dedup_window: int = 15):
        """
        Parameters
        ----------
        dedup_window : int
            Number of frames over which identical outputs are suppressed
            to prevent display flicker.
        """
        self._dedup_window = dedup_window
        self._history: deque = deque(maxlen=dedup_window)
        self._prev_output: str = ""
        self._frame_idx: int = 0

        log.info("TextGenerator ready (dedup_window=%d).", dedup_window)

    def generate(self, token_sequence: List[str],
                 frame_idx: int = 0) -> TextOutput:
        """
        Convert a token sequence to a TextOutput.

        Parameters
        ----------
        token_sequence : list of str  (from SemanticGraphState.token_sequence)
        frame_idx : int
        """
        self._frame_idx = frame_idx
        out = TextOutput(frame_idx=frame_idx)

        if not token_sequence:
            token_sequence = [T.NEUTRAL]

        out.raw_tokens = list(token_sequence)

        # Human-readable labels
        out.human_labels = [
            TOKEN_DESCRIPTIONS.get(tok, f"[{tok}]")
            for tok in token_sequence
        ]

        # Formatted string (raw)
        out.formatted_string = " ".join(token_sequence)

        # Deduplication check
        prev = self._prev_output
        out.is_new = (out.formatted_string != prev)
        self._prev_output = out.formatted_string
        self._history.append(out.formatted_string)

        return out

    def get_recent_history(self, n: int = 5) -> List[str]:
        """Return last n unique outputs."""
        seen = set()
        result = []
        for item in reversed(self._history):
            if item not in seen:
                seen.add(item)
                result.append(item)
            if len(result) >= n:
                break
        return list(reversed(result))

    def reset(self):
        self._history.clear()
        self._prev_output = ""
