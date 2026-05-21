"""
fusion_engine/semantic_priority_resolver.py
=============================================
Priority-Based Token Selection Layer

Resolves conflicts between simultaneously active tokens using
ISL linguistic priority rules. Real ISL grammar allows only
ONE primary sentence mode at a time, with a small number of
compatible modifiers.

Architecture position:
  temporal_memory → [semantic_priority_resolver] → grammar_engine

Priority hierarchy (derived from Zeshan 2004, ISL grammar):
  Level 1 — Sentence type (mutually exclusive, pick highest confidence)
  Level 2 — Discourse markers (at most 1)
  Level 3 — Modifiers (up to 2 compatible ones)

Example:
  Input:  [WH-Q, YN-Q, SURPRISE, EXCLAMATION, FOCUS, TOPIC_MARKER, EMPHASIS]
  Output: [WH-Q, FOCUS, EMPHASIS(strong)]   ← clean, linguistically valid

Suppression rules:
  WH-Q     suppresses  YN-Q (WH is more specific)
  NEGATION suppresses  AGREEMENT (contradiction)
  SURPRISE suppresses  EMPHASIS (override)
  DOUBT    suppresses  AGREEMENT (contradiction)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from config.config import LinguisticTokens

T = LinguisticTokens


# ── Priority tiers ──────────────────────────────────────────────────────────
# Tier 1: Sentence mode — ONLY ONE allowed
SENTENCE_MODES: List[str] = [
    T.QUESTION_WH,      # highest priority
    T.QUESTION_YN,
    T.NEGATION,
    T.CONDITIONAL,
    T.ASSERTION,
]

# Tier 2: Discourse markers — at most ONE
DISCOURSE_MARKERS: List[str] = [
    T.TOPIC_SHIFT,
    T.TOPIC_MARKER,
    T.FOCUS,
    T.BOUNDARY,
]

# Tier 3: Modifiers — up to TWO allowed
MODIFIERS: List[str] = [
    T.EMPHASIS_STRONG,
    T.EMPHASIS_MILD,
    T.SURPRISE,
    T.EXCLAMATION,
    T.DOUBT,
    T.UNCERTAINTY,
    T.AGREEMENT,
    T.DISAGREEMENT,
    T.CONFIRMATION,
]

# ── Hard suppression rules ───────────────────────────────────────────────────
# If token A is selected, token B is suppressed
SUPPRESSION_RULES: List[Tuple[str, str]] = [
    # WH-question is more specific — suppresses YN
    (T.QUESTION_WH,    T.QUESTION_YN),
    # Cannot negate and agree simultaneously
    (T.NEGATION,       T.AGREEMENT),
    (T.NEGATION,       T.CONFIRMATION),
    # Surprise overrides mild emphasis
    (T.SURPRISE,       T.EMPHASIS_MILD),
    # Exclamation overrides mild emphasis
    (T.EXCLAMATION,    T.EMPHASIS_MILD),
    # Strong emphasis overrides mild emphasis
    (T.EMPHASIS_STRONG,T.EMPHASIS_MILD),
    # Doubt contradicts agreement
    (T.DOUBT,          T.AGREEMENT),
    (T.UNCERTAINTY,    T.CONFIRMATION),
    # Topic shift overrides topic marker (shift is more specific)
    (T.TOPIC_SHIFT,    T.TOPIC_MARKER),
    # Disagreement contradicts agreement
    (T.DISAGREEMENT,   T.AGREEMENT),
]

# Build suppression lookup
_SUPPRESSION_MAP: Dict[str, Set[str]] = {}
for dominant, suppressed in SUPPRESSION_RULES:
    if dominant not in _SUPPRESSION_MAP:
        _SUPPRESSION_MAP[dominant] = set()
    _SUPPRESSION_MAP[dominant].add(suppressed)


# ── Compatibility rules ──────────────────────────────────────────────────────
# Which modifier pairs are linguistically compatible
COMPATIBLE_MODIFIER_PAIRS: Set[Tuple[str,str]] = {
    (T.SURPRISE,       T.EXCLAMATION),
    (T.EMPHASIS_STRONG,T.FOCUS),
    (T.DOUBT,          T.UNCERTAINTY),
    (T.AGREEMENT,      T.EMPHASIS_MILD),
    (T.NEGATION,       T.EMPHASIS_STRONG),
    (T.FOCUS,          T.EMPHASIS_STRONG),
}


@dataclass
class ResolvedTokens:
    """Output of the priority resolver — clean, conflict-free token list."""
    tokens: List[str]                   # final resolved list (max ~3 tokens)
    primary: Optional[str]              # the single sentence-mode token
    discourse: Optional[str]            # the single discourse marker
    modifiers: List[str]                # up to 2 modifiers
    suppressed: List[str]               # tokens that were removed
    resolution_log: List[str]           # human-readable explanation


class SemanticPriorityResolver:
    """
    Resolves token conflicts using ISL linguistic priority rules.

    Call resolve() with the raw confirmed token list and confidence
    scores. Returns a ResolvedTokens object with a clean, small,
    linguistically valid token set.
    """

    def resolve(self,
                raw_tokens: List[str],
                confidence_scores: Dict,
                graph_weights: Dict[str, float]) -> ResolvedTokens:
        """
        Parameters
        ----------
        raw_tokens        : confirmed token list from TemporalMemory
        confidence_scores : Dict[token_str, TokenConfidence]
        graph_weights     : Dict[token_str, float] from SFG

        Returns
        -------
        ResolvedTokens with at most 3-4 tokens
        """
        token_set   = set(raw_tokens) - {T.NEUTRAL}
        suppressed  = []
        log_lines   = []

        if not token_set:
            return ResolvedTokens(
                tokens=[T.NEUTRAL], primary=None,
                discourse=None, modifiers=[],
                suppressed=[], resolution_log=["No active tokens → NEUTRAL"]
            )

        def get_confidence(tok: str) -> float:
            cs = confidence_scores.get(tok)
            if cs:
                return cs.confidence
            return graph_weights.get(tok, 0.0)

        # ── Step 1: Select ONE sentence mode (highest confidence) ────────
        active_modes = [t for t in SENTENCE_MODES if t in token_set]
        primary = None

        if active_modes:
            # Sort by confidence descending
            active_modes.sort(key=lambda t: -get_confidence(t))
            primary = active_modes[0]
            # Suppress all other sentence modes
            for mode in active_modes[1:]:
                suppressed.append(mode)
                token_set.discard(mode)
                log_lines.append(
                    f"Suppressed {mode.split('(')[0]} "
                    f"(sentence mode conflict — {primary.split('(')[0]} takes priority)"
                )

        # ── Step 2: Apply suppression rules from selected primary ────────
        if primary:
            rules_suppressed = _SUPPRESSION_MAP.get(primary, set())
            for tok in list(rules_suppressed):
                if tok in token_set:
                    token_set.discard(tok)
                    suppressed.append(tok)
                    log_lines.append(
                        f"Suppressed {tok.split('(')[0]} "
                        f"(suppression rule: {primary.split('(')[0]} → suppress)"
                    )

        # ── Step 3: Apply suppression rules from all remaining tokens ────
        remaining = list(token_set)
        for tok in remaining:
            if tok in token_set:
                for supp in _SUPPRESSION_MAP.get(tok, set()):
                    if supp in token_set and supp != primary:
                        token_set.discard(supp)
                        suppressed.append(supp)
                        log_lines.append(
                            f"Suppressed {supp.split('(')[0]} "
                            f"(rule: {tok.split('(')[0]} suppresses it)"
                        )

        # ── Step 4: Select ONE discourse marker (highest confidence) ─────
        active_discourse = [t for t in DISCOURSE_MARKERS if t in token_set]
        discourse = None

        if active_discourse:
            active_discourse.sort(key=lambda t: -get_confidence(t))
            discourse = active_discourse[0]
            for d in active_discourse[1:]:
                token_set.discard(d)
                suppressed.append(d)
                log_lines.append(
                    f"Suppressed {d.split('(')[0]} "
                    f"(discourse conflict — {discourse.split('(')[0]} wins)"
                )

        # ── Step 5: Select UP TO 2 compatible modifiers ──────────────────
        active_mods = [t for t in MODIFIERS if t in token_set]
        active_mods.sort(key=lambda t: -get_confidence(t))

        selected_mods = []
        for mod in active_mods:
            if len(selected_mods) == 0:
                selected_mods.append(mod)
            elif len(selected_mods) == 1:
                # Check compatibility
                pair = (selected_mods[0], mod)
                pair_r = (mod, selected_mods[0])
                if pair in COMPATIBLE_MODIFIER_PAIRS or pair_r in COMPATIBLE_MODIFIER_PAIRS:
                    selected_mods.append(mod)
                else:
                    suppressed.append(mod)
                    log_lines.append(
                        f"Suppressed {mod.split('(')[0]} "
                        f"(incompatible with {selected_mods[0].split('(')[0]})"
                    )
            else:
                suppressed.append(mod)
                log_lines.append(
                    f"Suppressed {mod.split('(')[0]} (modifier limit reached)"
                )

        # ── Step 6: Build final token list ───────────────────────────────
        final_tokens = []
        if primary:   final_tokens.append(primary)
        if discourse: final_tokens.append(discourse)
        final_tokens.extend(selected_mods)

        if not final_tokens:
            final_tokens = [T.NEUTRAL]

        if suppressed:
            log_lines.insert(0,
                f"Resolved {len(raw_tokens)} → {len(final_tokens)} tokens "
                f"({len(suppressed)} suppressed)"
            )
        else:
            log_lines.insert(0, f"No conflicts — {len(final_tokens)} tokens retained")

        return ResolvedTokens(
            tokens         = final_tokens,
            primary        = primary,
            discourse      = discourse,
            modifiers      = selected_mods,
            suppressed     = suppressed,
            resolution_log = log_lines,
        )
