"""
fusion_engine/grammar_validator.py
=====================================
ISL Grammar Error Detection — Feature #9

Detects incorrect or incomplete ISL non-manual grammar structure.
Flags missing or contradictory markers that would make the
signing grammatically incorrect in ISL.

Rules derived from Zeshan (2004) ISL grammar documentation
and ISLRTC annotation guidelines.

Example warnings:
  ⚠ WH-Question detected but eyebrow raise is weak (< 0.4)
     → Missing required brow raise for WH-question marker
  ⚠ Negation + Agreement detected simultaneously
     → Contradictory markers: cannot negate and agree at once
  ✓ QUESTION(type=WH) structure is grammatically well-formed

Use cases: ISL education platforms, learning apps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from config.config import LinguisticTokens

T = LinguisticTokens


@dataclass
class GrammarWarning:
    code: str
    severity: str      # "ERROR" | "WARNING" | "INFO"
    message: str
    suggestion: str
    token_involved: str


@dataclass
class ValidationResult:
    is_valid: bool
    warnings: List[GrammarWarning]
    score: float           # 0-1 grammar quality score
    summary: str


# Grammar rules: (condition_fn, warning)
# condition_fn takes (tokens, feature_vector) -> bool (True = problem found)

class GrammarValidator:
    """
    Validates ISL non-manual grammar structure against linguistic rules.
    """

    def validate(self,
                 confirmed_tokens: List[str],
                 feature_vector: Dict[str, float],
                 confidence_scores: Dict) -> ValidationResult:

        tok = set(confirmed_tokens)
        warnings: List[GrammarWarning] = []

        # ── Rule 1: WH-question requires strong bilateral brow raise ──
        if T.QUESTION_WH in tok:
            brow = feature_vector.get("both_raised", 0.0)
            if brow < 0.50:
                warnings.append(GrammarWarning(
                    code    = "WH_BROW_WEAK",
                    severity = "WARNING",
                    message  = "WH-question detected but eyebrow raise is weak",
                    suggestion = "Raise both eyebrows more prominently for WH-questions",
                    token_involved = T.QUESTION_WH,
                ))

        # ── Rule 2: YN-question requires head nod ─────────────────────
        if T.QUESTION_YN in tok:
            nod = feature_vector.get("head_nod", 0.0)
            if nod < 0.40:
                warnings.append(GrammarWarning(
                    code    = "YN_NOD_MISSING",
                    severity = "WARNING",
                    message  = "YN-question detected but head nod is absent",
                    suggestion = "Add a slight forward nod for YN-questions",
                    token_involved = T.QUESTION_YN,
                ))

        # ── Rule 3: Cannot negate and agree simultaneously ────────────
        if T.NEGATION in tok and T.AGREEMENT in tok:
            warnings.append(GrammarWarning(
                code    = "CONTRADICT_NEG_AGREE",
                severity = "ERROR",
                message  = "Contradiction: NEGATION and AGREEMENT active simultaneously",
                suggestion = "These markers are mutually exclusive in ISL grammar",
                token_involved = T.NEGATION,
            ))

        # ── Rule 4: Negation requires head shake ──────────────────────
        if T.NEGATION in tok:
            shake = feature_vector.get("is_shaking", 0.0)
            if shake < 0.50:
                warnings.append(GrammarWarning(
                    code    = "NEG_SHAKE_WEAK",
                    severity = "WARNING",
                    message  = "Negation detected but head shake is weak",
                    suggestion = "Use a clearer lateral head shake for negation",
                    token_involved = T.NEGATION,
                ))

        # ── Rule 5: Emphasis requires sustained markers ───────────────
        if T.EMPHASIS_STRONG in tok:
            conf = confidence_scores.get(T.EMPHASIS_STRONG)
            if conf and conf.fused_pct < 60:
                warnings.append(GrammarWarning(
                    code    = "EMPHASIS_LOW_CONF",
                    severity = "INFO",
                    message  = "Strong emphasis has low confidence",
                    suggestion = "Hold the emphasis posture longer for clarity",
                    token_involved = T.EMPHASIS_STRONG,
                ))

        # ── Rule 6: Both question types simultaneously ────────────────
        if T.QUESTION_WH in tok and T.QUESTION_YN in tok:
            warnings.append(GrammarWarning(
                code    = "DUAL_QUESTION",
                severity = "WARNING",
                message  = "Both WH and YN question markers active",
                suggestion = "ISL uses distinct non-manual markers for each question type",
                token_involved = T.QUESTION_WH,
            ))

        # ── Rule 7: Surprise requires open mouth ─────────────────────
        if T.SURPRISE in tok:
            mouth = feature_vector.get("mouth_open", 0.0)
            if mouth < 0.30:
                warnings.append(GrammarWarning(
                    code    = "SURPRISE_MOUTH",
                    severity = "INFO",
                    message  = "Surprise marker detected but mouth is not sufficiently open",
                    suggestion = "Open mouth wider for surprise/exclamation markers",
                    token_involved = T.SURPRISE,
                ))

        # Compute score
        errors   = sum(1 for w in warnings if w.severity == "ERROR")
        warns    = sum(1 for w in warnings if w.severity == "WARNING")
        score    = max(0.0, 1.0 - errors * 0.4 - warns * 0.15)
        is_valid = errors == 0

        if not warnings:
            summary = "✓ Grammar structure is well-formed"
        elif errors > 0:
            summary = f"✗ {errors} grammar error(s) detected"
        else:
            summary = f"⚠ {warns} grammar warning(s)"

        return ValidationResult(
            is_valid = is_valid,
            warnings = warnings,
            score    = score,
            summary  = summary,
        )
