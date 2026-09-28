"""D6 grounding module v1 — equation/citation matching for grant proposals.

Adapted from D5 verifier_grounded_reward_v1.py concept, but uses simpler
regex + Jaccard matching (no sympy) appropriate for grant-proposal markdown.
D5's LaTeX sympy normalization is not suitable here.

Key functions:
  compute_grounding_bonus_spans()  — returns (char_start, char_end, bonus) spans
  check_grounding_saturation()     — Goodhart cliff guard
  load_gold()                      — load pre-extracted gold JSON
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Regex patterns — shared with extract_gold.py
# =============================================================================

# Display math: $$...$$, \[...\]
_DISPLAY_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]",
    re.DOTALL,
)
# Inline math: $...$ with ≥3 non-space characters (avoids spurious matches)
_INLINE_MATH_RE = re.compile(r"\$([^\$\n]{3,}?)\$")
# LaTeX environment
_LATEX_ENV_RE = re.compile(
    r"\\begin\{(equation|align|aligned|math)\}(.+?)\\end\{\1\}",
    re.DOTALL,
)

# Citation patterns (markdown + text)
_CITE_PATTERNS = [
    re.compile(r"\[@([\w-]+)\]"),                       # pandoc [@key]
    re.compile(r"\[(\d+)\]"),                           # numbered [1]
    re.compile(r"\(([\w]+ et al[.,] \d{4}[a-z]?)\)"),  # (Author et al. YYYY)
    re.compile(r"\(([\w]+ \d{4}[a-z]?)\)"),             # (Author YYYY)
]


# =============================================================================
# Normalization helpers
# =============================================================================

def _normalize_eq(text: str) -> str:
    """Strip whitespace and simple formatting for Jaccard comparison."""
    text = re.sub(r"\s+", "", text.lower())
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("{", "").replace("}", "")
    return text


def _math_tokens(normalized: str) -> set[str]:
    """Split normalized equation into tokens by operator boundaries."""
    parts = re.split(r"[+\-*/=<>(),^_\\]", normalized)
    return {p for p in parts if len(p) >= 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# =============================================================================
# Gold loading
# =============================================================================

def load_gold(gold_path: str | Path) -> dict:
    """Load pre-extracted gold equations/citations from gold.json."""
    path = Path(gold_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gold file not found at {path}. "
            "Run extract_gold.py first:\n"
            "  PYTHONPATH=src python -m co_scientist.d6_grant_proposal.extract_gold "
            "goal_domain=<domain> goal_name=<name>"
        )
    return json.loads(path.read_text())


# =============================================================================
# Equation extraction from candidate text
# =============================================================================

def _extract_eq_candidates(text: str) -> list[tuple[int, int, str]]:
    """Return list of (char_start, char_end, raw_text) for each equation in text."""
    candidates: list[tuple[int, int, str]] = []
    seen_spans: set[tuple[int, int]] = set()

    def _add(m: re.Match, group: int = 1):
        raw = m.group(group) or ""
        if raw.strip():
            span = (m.start(), m.end())
            if span not in seen_spans:
                seen_spans.add(span)
                candidates.append((m.start(), m.end(), raw.strip()))

    for m in _DISPLAY_MATH_RE.finditer(text):
        raw = m.group(1) or m.group(2) or ""
        if raw.strip():
            span = (m.start(), m.end())
            if span not in seen_spans:
                seen_spans.add(span)
                candidates.append((m.start(), m.end(), raw.strip()))
    for m in _INLINE_MATH_RE.finditer(text):
        _add(m)
    for m in _LATEX_ENV_RE.finditer(text):
        _add(m, group=2)
    return candidates


# =============================================================================
# Main grounding function
# =============================================================================

def compute_grounding_bonus_spans(
    plan_text: str,
    gold: dict,
    signal_set: tuple[str, ...] = ("equations", "citations"),
    per_eq_bonus: float = 0.1,
    per_citation_bonus: float = 0.05,
    eq_jaccard_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], dict]:
    """Compute per-char grounding bonus spans in plan_text.

    Returns:
      spans: list of (char_start, char_end, bonus_value) in plan_text coordinates
      diag:  diagnostic dict logged to metrics.jsonl
    """
    spans: list[tuple[int, int, float]] = []
    diag: dict = {"eq_hits": 0, "cite_hits": 0, "eq_details": [], "cite_details": []}

    # ---- Equation matching ----
    if "equations" in signal_set and gold.get("equations"):
        gold_eqs = gold["equations"]
        gold_normalized = [(eq["id"], _normalize_eq(eq["normalized"])) for eq in gold_eqs]
        gold_token_sets = [(eid, _math_tokens(norm)) for eid, norm in gold_normalized]

        cands = _extract_eq_candidates(plan_text)
        matched_gold_ids: set[str] = set()

        for c_start, c_end, c_raw in cands:
            c_norm = _normalize_eq(c_raw)
            c_tokens = _math_tokens(c_norm)
            if not c_tokens:
                continue
            for gold_id, gold_tokens in gold_token_sets:
                if gold_id in matched_gold_ids:
                    continue
                sim = _jaccard(c_tokens, gold_tokens)
                if sim >= eq_jaccard_threshold:
                    spans.append((c_start, c_end, per_eq_bonus))
                    matched_gold_ids.add(gold_id)
                    diag["eq_hits"] += 1
                    diag["eq_details"].append({
                        "gold_id": gold_id, "jaccard": round(sim, 3),
                        "candidate": c_raw[:80],
                    })
                    break  # one gold match per candidate

    # ---- Citation matching ----
    if "citations" in signal_set and gold.get("citations"):
        matched_gold_ids_cite: set[str] = set()

        for cite_entry in gold["citations"]:
            gold_id = cite_entry["id"]
            if gold_id in matched_gold_ids_cite:
                continue
            for alias in cite_entry.get("aliases", []):
                alias_stripped = alias.strip()
                if not alias_stripped:
                    continue
                idx = plan_text.find(alias_stripped)
                if idx >= 0:
                    end_idx = idx + len(alias_stripped)
                    spans.append((idx, end_idx, per_citation_bonus))
                    matched_gold_ids_cite.add(gold_id)
                    diag["cite_hits"] += 1
                    diag["cite_details"].append({
                        "gold_id": gold_id, "alias": alias_stripped,
                    })
                    break  # first alias match per citation

    logger.debug(
        "Grounding: %d eq_hits, %d cite_hits from %d eq and %d cite gold items",
        diag["eq_hits"], diag["cite_hits"],
        len(gold.get("equations", [])), len(gold.get("citations", [])),
    )
    return spans, diag


# =============================================================================
# Cliff guard
# =============================================================================

def check_grounding_saturation(
    bonus_token_count: int,
    total_solution_tokens: int,
    threshold: float = 0.85,
) -> bool:
    """Return True if grounding bonus covers > threshold of solution tokens.

    A True return means the grounding reward is likely Goodharting — the model
    has learned to produce surface-form matches that saturate the bonus without
    genuinely grounding its content. Caller should halve bonuses for this iter.
    """
    if total_solution_tokens == 0:
        return False
    return (bonus_token_count / total_solution_tokens) > threshold


# =============================================================================
# Token mask builder (given char spans in plan_text and position of plan in decoded)
# =============================================================================

def build_grounding_token_mask(
    *,
    tokenizer,
    decoded_full: str,
    sol_start_char: int,
    plan_text: str,
    spans: list[tuple[int, int, float]],
    sample_tokens: list[int],
) -> tuple["np.ndarray", int]:
    """Convert char spans (in plan_text) → per-token bonus array.

    Returns:
      bonus_array: shape (len(sample_tokens),) with bonus values at matching tokens
      bonus_token_count: number of tokens that received a non-zero bonus
    """
    import numpy as np
    bonus_array = np.zeros(len(sample_tokens), dtype=np.float32)
    bonus_token_count = 0

    for rel_start, rel_end, bonus_val in spans:
        abs_start = sol_start_char + rel_start
        abs_end = sol_start_char + rel_end
        if abs_end > len(decoded_full):
            continue
        try:
            tok_start = len(tokenizer.encode(decoded_full[:abs_start], add_special_tokens=False))
            tok_end = len(tokenizer.encode(decoded_full[:abs_end], add_special_tokens=False))
        except Exception:
            continue
        tok_start = max(0, min(tok_start, len(sample_tokens)))
        tok_end = max(tok_start, min(tok_end, len(sample_tokens)))
        if tok_end > tok_start:
            bonus_array[tok_start:tok_end] = bonus_val
            bonus_token_count += tok_end - tok_start

    return bonus_array, bonus_token_count
