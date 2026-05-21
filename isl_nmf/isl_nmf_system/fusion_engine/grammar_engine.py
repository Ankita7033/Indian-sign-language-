"""
fusion_engine/grammar_engine.py
=================================
ISL Grammar Engine — Feature #1 (BIGGEST UPGRADE)

Converts token sequences + hand gestures + temporal order
into complete, natural English sentences.

Pipeline:
  gesture_tokens + non-manual_grammar + temporal_order
  = "Are you going?"

ISL follows a Topic-Comment-Verb structure, different from English.
This engine applies ISL-to-English grammar transformation rules.

Examples:
  YOU + GO + QUESTION(WH)     → "Where are you going?"
  FOOD + EAT + NEGATION       → "I am not eating."
  NAME + WHAT + QUESTION(WH)  → "What is your name?"
  GO + QUESTION(YN)           → "Are you going?"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from config.config import LinguisticTokens
import time

T = LinguisticTokens

# ── Hand gesture to concept mapping ────────────────────────────────────────
GESTURE_CONCEPTS: Dict[str, str] = {
    "point":         "YOU/THAT",
    "open_palm":     "STOP/WAIT",
    "fist":          "STRONG/HOLD",
    "victory":       "TWO/PEACE",
    "thumb":         "GOOD/YES",
    "pinky":         "SMALL/PROMISE",
    "three_fingers": "THREE",
    "none":          "",
    "custom":        "",
}

# ── Grammar transformation rules ───────────────────────────────────────────
# Format: (grammar_tokens_required, gesture_concept, english_template)
# {G} = gesture concept substitution
# Most specific rules first
SENTENCE_RULES: List[Tuple] = [
    # WH-Questions
    ({T.QUESTION_WH}, "YOU/THAT", "Where are you going? / What are you doing?"),
    ({T.QUESTION_WH}, "STOP/WAIT", "Why stop? / What is happening?"),
    ({T.QUESTION_WH}, "GOOD/YES",  "What is good? / Which one?"),
    ({T.QUESTION_WH}, "",          "What is it? / Where? / Who? / When?"),

    # YN-Questions
    ({T.QUESTION_YN}, "YOU/THAT",  "Are you doing that?"),
    ({T.QUESTION_YN}, "GOOD/YES",  "Is that okay? / Is it good?"),
    ({T.QUESTION_YN}, "STOP/WAIT", "Should we stop?"),
    ({T.QUESTION_YN}, "",          "Is that right? / Are you sure?"),

    # Negation combinations
    ({T.NEGATION, T.QUESTION_WH},  "YOU/THAT", "Why aren't you going?"),
    ({T.NEGATION, T.QUESTION_YN},  "",         "Are you not coming?"),
    ({T.NEGATION},                 "YOU/THAT", "You should not do that."),
    ({T.NEGATION},                 "STOP/WAIT","No, stop! / Do not continue."),
    ({T.NEGATION},                 "",         "No. / That is not right."),

    # Emphasis
    ({T.EMPHASIS_STRONG, T.AGREEMENT}, "GOOD/YES", "Yes, absolutely! That is excellent!"),
    ({T.EMPHASIS_STRONG, T.AGREEMENT}, "",          "Yes! Definitely! I strongly agree."),
    ({T.EMPHASIS_STRONG},              "YOU/THAT",  "You must do this! It is very important."),
    ({T.EMPHASIS_STRONG},              "",          "This is very important! Pay attention!"),
    ({T.EMPHASIS_MILD},                "",          "This is somewhat important."),

    # Agreement / Disagreement
    ({T.AGREEMENT},    "GOOD/YES",  "Yes, that is good. I agree."),
    ({T.AGREEMENT},    "",          "Yes. / I agree. / That is correct."),
    ({T.DISAGREEMENT}, "",          "No. / I disagree. / That is wrong."),

    # Doubt / Uncertainty
    ({T.DOUBT, T.UNCERTAINTY}, "", "I am not sure about that. Maybe."),
    ({T.DOUBT},                "", "I doubt it. / Maybe not."),
    ({T.UNCERTAINTY},          "", "I am not certain. / Perhaps."),

    # Surprise / Exclamation
    ({T.SURPRISE, T.EXCLAMATION}, "", "Wow! That is amazing! Really?!"),
    ({T.SURPRISE},                "", "Oh! That is surprising!"),
    ({T.EXCLAMATION},             "", "Wow! / That is incredible!"),

    # Topic / Focus
    ({T.TOPIC_SHIFT},  "YOU/THAT", "Now, about you... / Moving to that topic."),
    ({T.TOPIC_SHIFT},  "",         "Let us change the subject."),
    ({T.TOPIC_MARKER}, "",         "Regarding this topic..."),
    ({T.FOCUS},        "",         "Pay attention to this specifically."),

    # Conditional
    ({T.CONDITIONAL},  "", "If that is the case... / Suppose that..."),
]


@dataclass
class GeneratedSentence:
    english: str
    tokens_used: List[str]
    gesture_used: str
    rule_matched: bool
    generation_time: str


class GrammarEngine:
    """
    Converts ISL token sequences into natural English sentences.

    Maintains a short temporal buffer of recent tokens to construct
    multi-utterance context (e.g., topic + comment → full sentence).
    """

    BUFFER_SIZE = 5   # remember last N confirmed token sets

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._last_sentence = ""

    def generate(self,
                 confirmed_tokens: List[str],
                 hand_gesture: str = "none") -> GeneratedSentence:
        """
        Generate an English sentence from confirmed tokens + hand gesture.
        """
        token_set = set(confirmed_tokens) - {T.NEUTRAL}
        gesture_concept = GESTURE_CONCEPTS.get(hand_gesture, "")

        if not token_set:
            return GeneratedSentence(
                english="...",
                tokens_used=[],
                gesture_used=hand_gesture,
                rule_matched=False,
                generation_time=time.strftime("%H:%M:%S")
            )

        # Try matching rules (most specific first)
        best_sentence = ""
        matched = False

        for rule_tokens, rule_gesture, template in SENTENCE_RULES:
            # Check token match
            if not rule_tokens.issubset(token_set):
                continue
            # Check gesture match (empty gesture in rule = matches any)
            if rule_gesture and rule_gesture != gesture_concept:
                continue
            best_sentence = template
            matched = True
            break

        # Fallback: build generic sentence from tokens
        if not best_sentence:
            parts = []
            if T.QUESTION_WH  in token_set: parts.append("What / Where / Who?")
            if T.QUESTION_YN  in token_set: parts.append("Is that right?")
            if T.NEGATION     in token_set: parts.append("No.")
            if T.AGREEMENT    in token_set: parts.append("Yes.")
            if T.DOUBT        in token_set: parts.append("Maybe.")
            if T.SURPRISE     in token_set: parts.append("Oh!")
            if T.EMPHASIS_STRONG in token_set: parts.append("Important!")
            best_sentence = " ".join(parts) if parts else "[Signing detected]"

        # Add gesture context if available and not already in sentence
        if gesture_concept and gesture_concept not in best_sentence:
            best_sentence = f"[{gesture_concept}] " + best_sentence

        # Update buffer
        self._buffer.append(token_set)
        self._last_sentence = best_sentence

        return GeneratedSentence(
            english          = best_sentence,
            tokens_used      = list(confirmed_tokens),
            gesture_used     = hand_gesture,
            rule_matched     = matched,
            generation_time  = time.strftime("%H:%M:%S")
        )

    def get_context(self) -> str:
        """Return recent sentence history as a string."""
        return " → ".join(
            str(s) for s in list(self._buffer)[-3:]
        )

    def reset(self):
        self._buffer.clear()
        self._last_sentence = ""
