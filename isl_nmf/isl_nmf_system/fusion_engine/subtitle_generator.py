"""
fusion_engine/subtitle_generator.py
=====================================
Real-Time Subtitle Generator

Converts ISL semantic token sequences into readable English subtitles.
Uses a rule-based grammar mapping aligned with ISL sentence structure.

Examples:
  QUESTION(type=WH)              → "What/Where/Who...?"
  QUESTION(type=YN) + NEGATION   → "Are you not...?"
  NEGATION(active)               → "[Negation]"
  EMPHASIS(strong) + AGREEMENT   → "Yes, definitely!"
  DOUBT + UNCERTAINTY            → "I'm not sure..."
  SURPRISE + EXCLAMATION         → "Wow!"
  TOPIC_SHIFT                    → "[New topic]"

The generator also maintains a rolling subtitle history
for display on screen.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
from config.config import LinguisticTokens

T = LinguisticTokens


# Grammar rule patterns: (token_set_subset, subtitle_text)
# Rules are matched in order — first match wins
GRAMMAR_RULES = [
    # Multi-token combinations (most specific first)
    ({T.QUESTION_WH,    T.NEGATION},         "Aren't you / Isn't it...?"),
    ({T.QUESTION_YN,    T.NEGATION},         "Are you not...?"),
    ({T.QUESTION_WH,    T.EMPHASIS_STRONG},  "What exactly...?!"),
    ({T.QUESTION_YN,    T.EMPHASIS_STRONG},  "Are you really...?"),
    ({T.EMPHASIS_STRONG, T.AGREEMENT},       "Yes, definitely!"),
    ({T.EMPHASIS_STRONG, T.EXCLAMATION},     "Absolutely!"),
    ({T.DOUBT,          T.UNCERTAINTY},      "I'm not sure..."),
    ({T.SURPRISE,       T.EXCLAMATION},      "Wow! / Really?!"),
    ({T.NEGATION,       T.DISAGREEMENT},     "No, that's wrong."),
    ({T.TOPIC_SHIFT,    T.TOPIC_MARKER},     "[New topic coming...]"),
    ({T.FOCUS,          T.EMPHASIS_STRONG},  "Pay attention to this!"),
    ({T.AGREEMENT,      T.CONFIRMATION},     "Yes, confirmed."),

    # Single tokens
    ({T.QUESTION_WH},    "What / Where / Who / When...?"),
    ({T.QUESTION_YN},    "Is it...? / Are you...?"),
    ({T.NEGATION},       "No / Not..."),
    ({T.EMPHASIS_STRONG},"[Strong emphasis]"),
    ({T.EMPHASIS_MILD},  "[Mild emphasis]"),
    ({T.AGREEMENT},      "Yes / I agree."),
    ({T.DISAGREEMENT},   "No / I disagree."),
    ({T.DOUBT},          "I doubt it / Maybe..."),
    ({T.UNCERTAINTY},    "I'm not sure..."),
    ({T.SURPRISE},       "Oh! / Really?"),
    ({T.EXCLAMATION},    "Wow! / Amazing!"),
    ({T.TOPIC_SHIFT},    "[Topic shift]"),
    ({T.TOPIC_MARKER},   "[Topic marker]"),
    ({T.CONDITIONAL},    "If... / Suppose..."),
    ({T.FOCUS},          "[Focus here]"),
    ({T.NEUTRAL},        ""),
]


@dataclass
class SubtitleEntry:
    text: str
    tokens: List[str]
    timestamp: str = ""


class SubtitleGenerator:
    """
    Converts confirmed token sequences to readable English subtitles.
    Maintains a rolling history for on-screen display.
    """

    MAX_HISTORY = 5

    def __init__(self):
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._last_text: str = ""

    def generate(self, tokens: List[str],
                 timestamp: str = "") -> SubtitleEntry:
        """
        Match tokens against grammar rules and return a subtitle.
        """
        token_set = set(tokens)

        # Try each rule in order
        matched_text = ""
        for rule_tokens, text in GRAMMAR_RULES:
            if rule_tokens.issubset(token_set):
                matched_text = text
                break

        # If no rule matched, build a generic description
        if not matched_text:
            active = [t for t in tokens if t != T.NEUTRAL]
            if active:
                matched_text = f"[{' + '.join(t.split('(')[0] for t in active[:3])}]"

        entry = SubtitleEntry(
            text      = matched_text,
            tokens    = list(tokens),
            timestamp = timestamp
        )

        # Only add to history if text changed
        if matched_text and matched_text != self._last_text:
            self._history.append(entry)
            self._last_text = matched_text

        return entry

    def get_history(self) -> List[SubtitleEntry]:
        return list(self._history)

    def reset(self):
        self._history.clear()
        self._last_text = ""
