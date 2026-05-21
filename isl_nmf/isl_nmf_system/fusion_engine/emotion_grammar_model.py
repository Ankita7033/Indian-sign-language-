"""
fusion_engine/emotion_grammar_model.py
========================================
Feature 18: Emotion-Grammar Interaction Model

Detects basic facial emotions and models how they interact
with ISL grammar tokens to produce enriched interpretations.

ISL emotions detected:
  HAPPY, SAD, ANGRY, CONFUSED, SURPRISED, NEUTRAL, FEARFUL

Interaction examples:
  QUESTION(WH) + CONFUSED  → "I don't understand. What is...?"
  NEGATION     + ANGRY      → "No! Absolutely not!"
  AGREEMENT    + HAPPY      → "Yes! Great! I agree!"
  DOUBT        + FEARFUL    → "I'm scared and not sure..."

Emotion is detected from:
  - Lip curvature (smile/frown)
  - Brow configuration (raised=surprise, furrowed=angry/confused)
  - Eye openness (wide=fear/surprise, narrowed=angry)
  - Combined landmark geometry
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from config.config import LinguisticTokens

T = LinguisticTokens


@dataclass
class EmotionResult:
    dominant: str        # primary emotion
    confidence: float
    scores: Dict[str, float]
    enriched_sentence: str   # grammar + emotion combined


# Emotion-grammar enrichment rules
ENRICHMENT_RULES = {
    (T.QUESTION_WH,     "CONFUSED"):   "I don't understand. What is...?",
    (T.QUESTION_WH,     "HAPPY"):      "Oh! What is this wonderful thing?",
    (T.QUESTION_YN,     "FEARFUL"):    "Are you okay? Is everything alright?",
    (T.QUESTION_YN,     "CONFUSED"):   "Wait, is that right?",
    (T.NEGATION,        "ANGRY"):      "No! Absolutely not!",
    (T.NEGATION,        "SAD"):        "No... I am sorry.",
    (T.AGREEMENT,       "HAPPY"):      "Yes! Great! I completely agree!",
    (T.AGREEMENT,       "NEUTRAL"):    "Yes, I agree.",
    (T.EMPHASIS_STRONG, "ANGRY"):      "This is VERY important! Listen!",
    (T.EMPHASIS_STRONG, "HAPPY"):      "This is wonderful! Pay attention!",
    (T.DOUBT,           "FEARFUL"):    "I am scared and not sure about this.",
    (T.DOUBT,           "CONFUSED"):   "I don't understand. Maybe?",
    (T.SURPRISE,        "HAPPY"):      "Wow! That is amazing!",
    (T.SURPRISE,        "FEARFUL"):    "Oh no! That is shocking!",
    (T.DISAGREEMENT,    "ANGRY"):      "No! That is completely wrong!",
    (T.TOPIC_SHIFT,     "NEUTRAL"):    "Let us move to the next point.",
}


class EmotionGrammarModel:
    """
    Detects emotion from face geometry and enriches grammar output.
    """

    def detect_emotion(self, fv: Dict[str, float]) -> EmotionResult:
        """
        Estimate emotion from feature vector using rule-based geometry.
        """
        scores: Dict[str, float] = {
            "HAPPY":     0.0,
            "SAD":       0.0,
            "ANGRY":     0.0,
            "CONFUSED":  0.0,
            "SURPRISED": 0.0,
            "FEARFUL":   0.0,
            "NEUTRAL":   0.4,
        }

        lip_spread   = fv.get("lip_spread", 0.0)
        mouth_open   = fv.get("mouth_open", 0.0)
        both_raised  = fv.get("both_raised", 0.0)
        furrowed     = fv.get("furrowed", 0.0)
        wide_eye     = fv.get("wide_eye", 0.0)
        mean_ear     = fv.get("mean_ear", 0.3)
        brow_asym    = fv.get("brow_raise_one", 0.0)
        face_stable  = fv.get("face_stable", 1.0)

        # HAPPY: lip spread must be significantly higher than baseline
        scores["HAPPY"]    = max(0.0, lip_spread - 0.4) * 2.0 + (mouth_open * 0.2)

        # SAD: lip spread must be significantly lower than baseline
        scores["SAD"]      = max(0.0, (0.3 - lip_spread) * 1.5)

        # ANGRY: furrowed + narrowed eyes
        scores["ANGRY"]    = furrowed * 0.7 + max(0.0, (0.25 - mean_ear) * 2) * 0.3

        # CONFUSED: asymmetric brow + head tilt
        scores["CONFUSED"] = brow_asym * 0.6 + fv.get("head_tilt", 0) * 0.4

        # SURPRISED: MUST have BOTH raised brows AND wide eyes, plus open mouth helps
        scores["SURPRISED"] = min(both_raised, wide_eye) * 0.7 + mouth_open * 0.3

        # FEARFUL: wide eyes + raised brows + mouth open (but less spread than surprise)
        scores["FEARFUL"]  = min(wide_eye, both_raised) * 0.6 * max(0.0, 1 - lip_spread)

        # Neutral: stable face, slightly higher base threshold to prevent noisy spikes
        scores["NEUTRAL"]  = max(0.4, face_stable * 0.7)

        dominant = max(scores, key=scores.get)
        conf     = min(scores[dominant], 1.0)

        return EmotionResult(
            dominant         = dominant,
            confidence       = conf,
            scores           = scores,
            enriched_sentence = "",
        )

    def enrich(self, grammar_tokens: List[str],
               emotion: EmotionResult,
               base_sentence: str) -> str:
        """
        Combine grammar + emotion to produce enriched sentence.
        """
        for tok in grammar_tokens:
            key = (tok, emotion.dominant)
            if key in ENRICHMENT_RULES:
                return ENRICHMENT_RULES[key]
        # No rule matched — append emotion qualifier
        if emotion.dominant != "NEUTRAL" and emotion.confidence > 0.55:
            return f"{base_sentence} [{emotion.dominant.lower()}]"
        return base_sentence
