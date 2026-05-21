"""
fusion_engine/isl_interpreter.py
==================================
Full ISL Interpreter — fuses hand gestures with non-manual grammar tokens
to produce complete ISL sentence interpretations.

Hand gesture + Non-manual grammar = Complete ISL meaning

Examples:
  point (hand) + QUESTION(type=WH)   → "Where is...? / Who is...?"
  open_palm (hand) + NEGATION        → "No / Stop"
  victory (hand) + AGREEMENT         → "Yes, two / Okay!"
  fist (hand) + EMPHASIS(strong)     → "Strong / Powerful"
  point (hand) + TOPIC_SHIFT         → "That one, new topic"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from config.config import LinguisticTokens

T = LinguisticTokens

# Fusion rules: (hand_gesture, grammar_tokens_subset) -> interpretation
FUSION_RULES: List[Tuple] = [
    # Hand gesture + grammar token → full meaning
    ("point",       {T.QUESTION_WH},        "Where is...? / Who is...?"),
    ("point",       {T.QUESTION_YN},        "Is that...?"),
    ("open_palm",   {T.NEGATION},           "No! / Stop!"),
    ("open_palm",   {T.QUESTION_YN},        "Is it okay?"),
    ("open_palm",   {T.AGREEMENT},          "Okay / Agreed"),
    ("fist",        {T.EMPHASIS_STRONG},    "Strong / Powerful / Important!"),
    ("fist",        {T.NEGATION},           "Absolutely not!"),
    ("victory",     {T.AGREEMENT},          "Yes! / Great!"),
    ("victory",     {T.QUESTION_YN},        "Are these two okay?"),
    ("thumb",       {T.AGREEMENT},          "Good / Thumbs up!"),
    ("thumb",       {T.EMPHASIS_STRONG},    "Excellent!"),
    ("pinky",       {T.QUESTION_WH},        "What is this?"),
    ("three_fingers",{T.TOPIC_MARKER},      "Three things to note..."),
    ("open_palm",   {T.TOPIC_SHIFT},        "Moving to next point..."),
    ("point",       {T.EMPHASIS_STRONG},    "That one, specifically!"),
    ("fist",        {T.DOUBT},              "I strongly doubt it"),
    ("open_palm",   {T.SURPRISE},           "Oh wow!"),

    # Hand only (no grammar signal)
    ("point",       set(),   "That / There / Him / Her"),
    ("open_palm",   set(),   "Stop / Wait / Hello"),
    ("fist",        set(),   "Strong / Hold / Grasp"),
    ("victory",     set(),   "Two / Victory / Peace"),
    ("thumb",       set(),   "Good / Okay / Approve"),
    ("pinky",       set(),   "Small / Little / Promise"),
    ("three_fingers",set(),  "Three / Several"),
    ("none",        set(),   ""),
]


@dataclass
class ISLInterpretation:
    hand_gesture: str
    grammar_tokens: List[str]
    full_interpretation: str
    confidence_pct: int
    is_complete: bool   # True = hand + grammar both present


class ISLInterpreter:
    """
    Combines hand gesture labels with confirmed grammar tokens
    to produce complete ISL sentence-level interpretations.
    """

    def interpret(self,
                  hand_gesture: str,
                  confirmed_tokens: List[str],
                  confidence_scores: Dict) -> ISLInterpretation:
        """
        Match hand + grammar to the fusion rules.
        """
        token_set = set(confirmed_tokens) - {T.NEUTRAL}
        best_text = ""
        best_is_complete = False

        for rule_gesture, rule_tokens, rule_text in FUSION_RULES:
            # Gesture must match
            if rule_gesture != "none" and rule_gesture != hand_gesture:
                continue
            # Grammar tokens must be subset of active tokens
            if rule_tokens and not rule_tokens.issubset(token_set):
                continue
            best_text        = rule_text
            best_is_complete = bool(rule_tokens) and hand_gesture != "none"
            break

        # If no rule matched but we have grammar tokens
        if not best_text and token_set:
            labels = [t.split("(")[0] for t in confirmed_tokens[:2]]
            best_text = f"[{' + '.join(labels)}]"

        # Confidence: average of available token confidences
        conf_values = [
            v.fused_pct for v in confidence_scores.values()
        ] if confidence_scores else [50]
        avg_conf = int(sum(conf_values) / len(conf_values)) if conf_values else 50

        return ISLInterpretation(
            hand_gesture        = hand_gesture,
            grammar_tokens      = confirmed_tokens,
            full_interpretation = best_text,
            confidence_pct      = avg_conf,
            is_complete         = best_is_complete,
        )
