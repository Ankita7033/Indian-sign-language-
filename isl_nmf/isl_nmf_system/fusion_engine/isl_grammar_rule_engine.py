"""
fusion_engine/isl_grammar_rule_engine.py
==========================================
Feature 19: ISL Grammar Rule Engine

A formal rule-based engine for ISL non-manual grammar.
Encodes documented ISL grammatical rules from linguistic
literature (Zeshan 2004, ISLRTC 2019) as executable logic.

Rules cover:
  1. Question formation (WH and YN)
  2. Negation structure
  3. Topic-comment structure
  4. Conditional clauses
  5. Agreement/disagreement marking
  6. Emphasis and focus
  7. Temporal reference (past/present/future via NMS)

Each rule has:
  - Required non-manual markers (NMS)
  - Optional co-occurring markers
  - Prohibited markers (contradictions)
  - Linguistic description
  - Example sentence structure

Output: structured grammatical parse + rule activation log
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from config.config import LinguisticTokens

T = LinguisticTokens


@dataclass
class GrammarRule:
    rule_id: str
    name: str
    description: str
    required_markers: Set[str]
    optional_markers: Set[str]
    prohibited_markers: Set[str]
    sentence_structure: str     # ISL structure template
    english_gloss: str          # English equivalent


@dataclass
class GrammarParse:
    matched_rules: List[GrammarRule]
    active_structure: str
    gloss: str
    violations: List[str]
    grammaticality_score: float   # 0-1
    rule_log: List[str]


# ISL Grammar Rule Definitions
ISL_GRAMMAR_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id   = "ISL-WH-Q",
        name      = "WH-Question Formation",
        description = "WH-questions in ISL require bilateral eyebrow raise "
                      "sustained throughout the question. Head may tilt forward.",
        required_markers  = {T.QUESTION_WH},
        optional_markers  = {T.FOCUS, T.EMPHASIS_MILD},
        prohibited_markers= {T.NEGATION, T.AGREEMENT},
        sentence_structure = "TOPIC [NMM: brow-raise] QUESTION-WORD",
        english_gloss      = "What/Where/Who/When/Why/How?"
    ),
    GrammarRule(
        rule_id   = "ISL-YN-Q",
        name      = "Yes/No Question Formation",
        description = "YN-questions use head nod + raised brows or forward head tilt.",
        required_markers  = {T.QUESTION_YN},
        optional_markers  = {T.AGREEMENT, T.EMPHASIS_MILD},
        prohibited_markers= {T.NEGATION, T.DISAGREEMENT},
        sentence_structure = "PREDICATE [NMM: head-nod, brow-raise]",
        english_gloss      = "Is it...? / Are you...?"
    ),
    GrammarRule(
        rule_id   = "ISL-NEG",
        name      = "Negation Marking",
        description = "Negation uses lateral head shake throughout the negated predicate. "
                      "Brow may furrow. Cannot co-occur with agreement.",
        required_markers  = {T.NEGATION},
        optional_markers  = {T.EMPHASIS_STRONG, T.DISAGREEMENT},
        prohibited_markers= {T.AGREEMENT, T.CONFIRMATION},
        sentence_structure = "SUBJECT PREDICATE [NMM: head-shake, brow-furrow]",
        english_gloss      = "NOT / No / Do not"
    ),
    GrammarRule(
        rule_id   = "ISL-TOPIC",
        name      = "Topic-Comment Structure",
        description = "Topics are marked with brow raise and slight head tilt. "
                      "The comment follows after the topic NMS.",
        required_markers  = {T.TOPIC_MARKER},
        optional_markers  = {T.TOPIC_SHIFT, T.FOCUS},
        prohibited_markers= set(),
        sentence_structure = "TOPIC [NMM: brow-raise, head-tilt] COMMENT",
        english_gloss      = "As for [topic]... [comment]"
    ),
    GrammarRule(
        rule_id   = "ISL-COND",
        name      = "Conditional Clause",
        description = "Conditionals use head tilt + brow raise on the condition, "
                      "then neutral or assertion NMS on the consequence.",
        required_markers  = {T.CONDITIONAL},
        optional_markers  = {T.TOPIC_SHIFT, T.FOCUS},
        prohibited_markers= {T.NEGATION},
        sentence_structure = "IF-CLAUSE [NMM: brow-raise, head-tilt] THEN-CLAUSE",
        english_gloss      = "If... then..."
    ),
    GrammarRule(
        rule_id   = "ISL-EMPH",
        name      = "Emphasis Marking",
        description = "Strong emphasis uses nod + wide eyes. "
                      "Mild emphasis uses single brow raise or slight nod.",
        required_markers  = {T.EMPHASIS_STRONG},
        optional_markers  = {T.FOCUS, T.EXCLAMATION},
        prohibited_markers= {T.DOUBT, T.UNCERTAINTY},
        sentence_structure = "SIGN [NMM: nod, wide-eyes, shoulder-raise]",
        english_gloss      = "Very / Really / Important!"
    ),
    GrammarRule(
        rule_id   = "ISL-AGREE",
        name      = "Agreement / Affirmation",
        description = "Agreement uses repeated downward nod. "
                      "Cannot co-occur with negation or disagreement.",
        required_markers  = {T.AGREEMENT},
        optional_markers  = {T.EMPHASIS_MILD, T.CONFIRMATION},
        prohibited_markers= {T.NEGATION, T.DISAGREEMENT, T.DOUBT},
        sentence_structure = "SIGN [NMM: head-nod-repeated]",
        english_gloss      = "Yes / I agree / That is correct"
    ),
    GrammarRule(
        rule_id   = "ISL-DOUBT",
        name      = "Doubt / Uncertainty Marking",
        description = "Doubt uses shoulder shrug + furrowed brows. "
                      "Conveys uncertainty about truth or identity.",
        required_markers  = {T.DOUBT},
        optional_markers  = {T.UNCERTAINTY, T.CONDITIONAL},
        prohibited_markers= {T.AGREEMENT, T.EMPHASIS_STRONG},
        sentence_structure = "SIGN [NMM: shrug, brow-furrow]",
        english_gloss      = "Maybe / I don't know / Possibly"
    ),
]

# Build lookup
_RULE_MAP: Dict[str, GrammarRule] = {r.rule_id: r for r in ISL_GRAMMAR_RULES}


class ISLGrammarRuleEngine:
    """
    Formal ISL grammar rule matcher and parse generator.

    Given active tokens, returns which grammar rules are triggered,
    whether any violations exist, and the active grammatical structure.
    """

    def parse(self, active_tokens: List[str],
              feature_vector: Dict[str, float]) -> GrammarParse:
        token_set     = set(active_tokens)
        matched       = []
        violations    = []
        log_lines     = []

        for rule in ISL_GRAMMAR_RULES:
            # Check if required markers are present
            if not rule.required_markers.issubset(token_set):
                continue

            # Check for prohibited markers
            proh = rule.prohibited_markers & token_set
            if proh:
                violations.append(
                    f"Rule {rule.rule_id}: prohibited marker(s) present: "
                    f"{[t.split('(')[0] for t in proh]}"
                )
                log_lines.append(f"⚠ {rule.name} violated by {proh}")
            else:
                matched.append(rule)
                opt_present = rule.optional_markers & token_set
                log_lines.append(
                    f"✓ {rule.name} matched"
                    + (f" + optional: {[t.split('(')[0] for t in opt_present]}"
                       if opt_present else "")
                )

        # Build active structure
        if matched:
            # Primary rule = first matched (by priority order)
            primary = matched[0]
            structure = primary.sentence_structure
            gloss     = primary.english_gloss
        else:
            structure = "NEUTRAL [NMM: none]"
            gloss     = "[Neutral — no grammar rule active]"

        # Grammaticality score
        score = max(0.0, 1.0 - len(violations) * 0.3)
        if matched:
            score = min(score + 0.2 * len(matched), 1.0)

        return GrammarParse(
            matched_rules        = matched,
            active_structure     = structure,
            gloss                = gloss,
            violations           = violations,
            grammaticality_score = score,
            rule_log             = log_lines,
        )

    def get_rule_descriptions(self) -> List[str]:
        return [f"{r.rule_id}: {r.name} — {r.english_gloss}"
                for r in ISL_GRAMMAR_RULES]
