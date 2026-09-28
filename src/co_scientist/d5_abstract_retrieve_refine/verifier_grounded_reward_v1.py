"""D5 v9 Layer 1 verifier-grounded reward — sparse per-token bonus on
oracle-fidelity hits (equations + citations + optional claim verification).

Per ABC verdict: H16-2 (incentive misalignment) is dominant fault. SDPO advantage
rewards critique-conformity, not oracle-fidelity. Layer 1 fix: add a
verifier-grounded reward as sparse per-token bonus on plan-text spans where
gold equations / citations are detected.

Bonus design (tier check on G+ peak plans showed sympy parse rate 26.7%, below
30% threshold; meaningful equations like J_RS with measure subscripts all in
tier (c)). Per-eq bonus reduced to 0.1 (Goodhart caution) AND fuzzy key-token
fallback added as secondary signal to recover tier (c) misses.

Three signals:
  - "equations": tier_a sympy.simplify match → tier_b normalize+sympify →
    tier_c fuzzy key_token match (≥fuzzy_min_tokens of 5 distinctive LaTeX
    fragments in same sentence)
  - "citations": regex extract author-year + cross-check against gold aliases
  - "claims": reuse seven_signal_reward.programmatic_claim_verification gate
    (optional, requires injected claims_extractor_fn)

Used by `train_mu_v9_grounded.py`. Pure CPU sync. Returns per-token bonus
spans for caller to add to advantage tensor before forward_backward.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lazy imports — sympy/latex2sympy only loaded if equations signal active
_SYMPY: Any = None
_LATEX2SYMPY: Any = None


def _ensure_sympy():
    global _SYMPY, _LATEX2SYMPY
    if _SYMPY is None:
        import sympy as _s
        _SYMPY = _s
    if _LATEX2SYMPY is None:
        try:
            from latex2sympy2_extended import latex2sympy as _l
            _LATEX2SYMPY = _l
        except ImportError:
            logger.warning("latex2sympy2_extended not installed; tier_a disabled")
            _LATEX2SYMPY = False  # sentinel
    return _SYMPY, _LATEX2SYMPY


# ---------------------------------------------------------------------------
# Equation extraction from plan text
# ---------------------------------------------------------------------------

_EQ_PATTERNS = [
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),  # display math
    re.compile(r'(?<!\$)\$(?!\$)([^\$\n]{5,500})\$(?!\$)'),  # inline math
    re.compile(r'\\\[(.+?)\\\]', re.DOTALL),  # display brackets
    re.compile(r'```math\n(.+?)\n```', re.DOTALL),  # markdown math fence
]


@dataclass
class CandidateEq:
    text: str  # raw equation text
    char_start: int  # absolute char position in plan
    char_end: int


def extract_eq_candidates(plan_text: str) -> list[CandidateEq]:
    """Extract candidate equations from plan text with char offsets."""
    seen_spans: set[tuple[int, int]] = set()
    candidates: list[CandidateEq] = []
    for pat in _EQ_PATTERNS:
        for m in pat.finditer(plan_text):
            eq = m.group(1).strip()
            # Filter: must look like equation (= or \nabla or \frac, length >10)
            if not (('=' in eq or '\\nabla' in eq or '\\frac' in eq) and len(eq) > 10):
                continue
            span = (m.start(1), m.end(1))
            if span in seen_spans:
                continue
            seen_spans.add(span)
            candidates.append(CandidateEq(text=eq, char_start=m.start(1), char_end=m.end(1)))
    return candidates


# ---------------------------------------------------------------------------
# Equation verification — tiered match
# ---------------------------------------------------------------------------

@dataclass
class EqMatch:
    candidate: CandidateEq
    gold_id: str
    tier: str  # "a" sympy_strict | "b" sympy_normalized | "c" fuzzy_key_tokens
    tokens_matched: int  # for tier_c only


def _normalize_for_tier_b(s: str) -> str:
    """Aggressive normalize for tier_b sympy parse."""
    s = s.replace('\\frac', '').replace('\\left', '').replace('\\right', '')
    s = s.replace('\\mathbb{E}', 'E').replace('\\mathbb{R}', 'R')
    s = s.replace('\\beta', 'beta').replace('\\theta', 'theta').replace('\\eta', 'eta')
    s = s.replace('\\pi', 'pi').replace('\\nabla', 'D').replace('\\sum', 'sum')
    s = s.replace('\\log', 'log').replace('\\exp', 'exp').replace('\\sqrt', 'sqrt')
    s = s.replace('^', '**').replace('{', '(').replace('}', ')')
    s = re.sub(r'\\[a-zA-Z]+', '', s)  # strip remaining LaTeX commands
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _sentence_around(plan_text: str, char_pos: int, max_chars: int = 400) -> str:
    """Return ±max_chars window around char_pos (sentence-ish context)."""
    start = max(0, char_pos - max_chars // 2)
    end = min(len(plan_text), char_pos + max_chars // 2)
    return plan_text[start:end]


def verify_equations(
    plan_text: str,
    gold_equations: list[dict],
) -> list[EqMatch]:
    """For each plan equation candidate, find best gold match across tiers."""
    sympy_mod, latex2sympy = _ensure_sympy()
    candidates = extract_eq_candidates(plan_text)
    matches: list[EqMatch] = []

    # Pre-parse gold sympy_aliases
    gold_parsed: dict[str, list] = {}  # gold_id -> list of sympy expressions
    for gold in gold_equations:
        parsed: list = []
        for alias in gold.get("sympy_aliases", []):
            try:
                parsed.append(sympy_mod.sympify(alias))
            except Exception:
                pass
        gold_parsed[gold["id"]] = parsed

    matched_gold_ids: set[str] = set()  # one gold per match max

    for cand in candidates:
        # Tier (a) — latex2sympy parse + simplify against gold sympy_aliases
        tier_a_hit = None
        if latex2sympy and latex2sympy is not False:
            try:
                cand_sym = latex2sympy(cand.text)
                for gold in gold_equations:
                    if gold["id"] in matched_gold_ids:
                        continue
                    for gold_sym in gold_parsed.get(gold["id"], []):
                        try:
                            if sympy_mod.simplify(cand_sym - gold_sym) == 0:
                                tier_a_hit = gold["id"]
                                break
                        except Exception:
                            pass
                    if tier_a_hit:
                        break
            except Exception:
                pass
        if tier_a_hit:
            matches.append(EqMatch(candidate=cand, gold_id=tier_a_hit, tier="a", tokens_matched=0))
            matched_gold_ids.add(tier_a_hit)
            continue

        # Tier (b) — normalize + sympify match
        tier_b_hit = None
        try:
            cand_norm = sympy_mod.sympify(_normalize_for_tier_b(cand.text))
            for gold in gold_equations:
                if gold["id"] in matched_gold_ids:
                    continue
                for gold_sym in gold_parsed.get(gold["id"], []):
                    try:
                        if sympy_mod.simplify(cand_norm - gold_sym) == 0:
                            tier_b_hit = gold["id"]
                            break
                    except Exception:
                        pass
                if tier_b_hit:
                    break
        except Exception:
            pass
        if tier_b_hit:
            matches.append(EqMatch(candidate=cand, gold_id=tier_b_hit, tier="b", tokens_matched=0))
            matched_gold_ids.add(tier_b_hit)
            continue

        # Tier (c) — fuzzy key_tokens match in surrounding context
        ctx = _sentence_around(plan_text, cand.char_start, max_chars=500).lower()
        tier_c_hit = None
        tier_c_count = 0
        for gold in gold_equations:
            if gold["id"] in matched_gold_ids:
                continue
            tokens = [t.lower() for t in gold.get("key_tokens", [])]
            n_hit = sum(1 for t in tokens if t in ctx)
            min_required = gold.get("fuzzy_min_tokens", 3)
            if n_hit >= min_required and n_hit > tier_c_count:
                tier_c_hit = gold["id"]
                tier_c_count = n_hit
        if tier_c_hit:
            matches.append(EqMatch(candidate=cand, gold_id=tier_c_hit, tier="c", tokens_matched=tier_c_count))
            matched_gold_ids.add(tier_c_hit)

    return matches


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

@dataclass
class CitationMatch:
    gold_id: str
    char_start: int
    char_end: int
    matched_alias: str


_CITE_RE = re.compile(r'\b([A-Z][a-z]+)(?:\s+(?:et al\.?|and\s+[A-Z][a-z]+))?\s*[(,]?\s*(\d{4})')


def verify_citations(
    plan_text: str,
    gold_citations: list[dict],
    year_max: int = 2026,
) -> list[CitationMatch]:
    """Find gold-citation hits in plan via author-year + alias regex."""
    matches: list[CitationMatch] = []
    matched_gold_ids: set[str] = set()

    # Build alias lookup: lowercase alias → gold_id
    alias_lookup: dict[str, str] = {}
    for gold in gold_citations:
        for alias in gold.get("aliases", []):
            alias_lookup[alias.lower()] = gold["id"]

    # Strategy 1: alias substring match in plan (case-insensitive)
    plan_lower = plan_text.lower()
    for alias_low, gold_id in alias_lookup.items():
        if gold_id in matched_gold_ids:
            continue
        idx = plan_lower.find(alias_low)
        if idx >= 0:
            matches.append(CitationMatch(
                gold_id=gold_id, char_start=idx, char_end=idx + len(alias_low),
                matched_alias=alias_low,
            ))
            matched_gold_ids.add(gold_id)

    # Strategy 2: author-year regex; filter year ≤ year_max (block fabricated 2027+)
    for m in _CITE_RE.finditer(plan_text):
        author, year_str = m.group(1), m.group(2)
        try:
            year = int(year_str)
        except ValueError:
            continue
        if year > year_max:
            continue  # fabricated future year
        for gold in gold_citations:
            if gold["id"] in matched_gold_ids:
                continue
            if gold["author"].lower() == author.lower() and abs(gold["year"] - year) <= 1:
                matches.append(CitationMatch(
                    gold_id=gold["id"], char_start=m.start(), char_end=m.end(),
                    matched_alias=f"{author} {year}",
                ))
                matched_gold_ids.add(gold["id"])
                break

    return matches


# ---------------------------------------------------------------------------
# Char-span → token-id mapping
# ---------------------------------------------------------------------------

def char_spans_to_token_spans(
    plan_text: str,
    char_spans: list[tuple[int, int]],
    tokenizer,
    plan_token_offset: int = 0,
) -> list[tuple[int, int]]:
    """Map char-spans of plan_text to token-id ranges in the generated sequence.

    Uses tokenizer.encode on prefix to map char position to token index.
    `plan_token_offset` shifts the resulting token positions by the count of
    prompt tokens (so caller can index into the full per-token advantage vector).
    """
    if not char_spans:
        return []
    token_spans: list[tuple[int, int]] = []
    for c_start, c_end in char_spans:
        try:
            prefix_tokens = tokenizer.encode(plan_text[:c_start])
            span_tokens = tokenizer.encode(plan_text[:c_end])
            t_start = plan_token_offset + len(prefix_tokens)
            t_end = plan_token_offset + len(span_tokens)
            if t_end > t_start:
                token_spans.append((t_start, t_end))
        except Exception as e:
            logger.warning("char→token mapping failed for span (%d,%d): %s", c_start, c_end, e)
    return token_spans


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def load_gold(gold_path: str | Path) -> dict:
    return json.loads(Path(gold_path).read_text())


def compute_grounding_reward(
    plan_text: str,
    gold: dict,  # full v9_gold_equations.json content
    signal_set: tuple[str, ...] = ("equations", "citations", "claims"),
    per_eq_bonus: float = 0.1,
    per_citation_bonus: float = 0.05,
    claims_extractor_fn: Callable[[str], dict | None] | None = None,
) -> dict:
    """Compute grounding reward per plan.

    Returns dict with:
      - score: float (sum of bonuses, normalized)
      - char_bonus_spans: list[(char_start, char_end, bonus)]  — caller maps to tokens
      - per_signal: {"equations": n_hits, "citations": n_hits, "claims": float}
      - diag: per-equation/citation match details for buffer.jsonl logging
    """
    char_bonus_spans: list[tuple[int, int, float]] = []
    per_signal: dict[str, Any] = {}
    diag: dict[str, Any] = {"eq_matches": [], "cite_matches": [], "claim_score": None}

    if "equations" in signal_set:
        eq_matches = verify_equations(plan_text, gold["equations"])
        per_signal["equations"] = len(eq_matches)
        for em in eq_matches:
            char_bonus_spans.append((em.candidate.char_start, em.candidate.char_end, per_eq_bonus))
            diag["eq_matches"].append({
                "gold_id": em.gold_id, "tier": em.tier,
                "candidate_text": em.candidate.text[:120], "tokens_matched": em.tokens_matched,
            })

    if "citations" in signal_set:
        cite_matches = verify_citations(plan_text, gold["citations"])
        per_signal["citations"] = len(cite_matches)
        for cm in cite_matches:
            char_bonus_spans.append((cm.char_start, cm.char_end, per_citation_bonus))
            diag["cite_matches"].append({
                "gold_id": cm.gold_id, "matched_alias": cm.matched_alias,
            })

    if "claims" in signal_set and claims_extractor_fn is not None:
        # Inject extractor: takes plan_text, returns parsed claims dict
        try:
            claims_parsed = claims_extractor_fn(plan_text)
            if claims_parsed:
                # Reuse seven_signal_reward.programmatic_claim_verification
                from co_scientist.shared.seven_signal_reward import programmatic_claim_verification
                cv_score, _cv_diag = programmatic_claim_verification(claims_parsed)
                per_signal["claims"] = float(cv_score)
                diag["claim_score"] = float(cv_score)
                # Note: claims contributes to overall score but not per-token bonus
                # (it's a plan-level signal; contributes to score field only)
        except Exception as e:
            logger.warning("claims extraction failed: %s", e)
            per_signal["claims"] = 0.0

    # Aggregate score: sum of bonuses + claims (claims adds 0.1 if =1.0)
    eq_pts = per_signal.get("equations", 0) * per_eq_bonus
    cite_pts = per_signal.get("citations", 0) * per_citation_bonus
    claim_pts = per_signal.get("claims", 0.0) * 0.1  # claims contribute up to 0.1
    score = eq_pts + cite_pts + claim_pts

    return {
        "score": float(score),
        "char_bonus_spans": char_bonus_spans,
        "per_signal": per_signal,
        "diag": diag,
    }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    REPO = Path(__file__).resolve().parents[3]
    gold = load_gold(REPO / "projects/d5_abstract_retrieve_refine/dataset/v9_gold_equations.json")
    print(f"Loaded gold: {len(gold['equations'])} equations + {len(gold['citations'])} citations")

    # Test on G+ peak plan
    import sys
    sys.path.insert(0, str(REPO / "src"))
    GPLUS_BUF = REPO / "projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus/buffer.jsonl"
    plans = []
    with open(GPLUS_BUF) as f:
        for line in f:
            r = json.loads(line)
            if r["iter"] in (4, 5):
                plans.append(r["plan_text"])
    print(f"Test on {len(plans)} G+ peak plans (iter 4-5)")

    total = {"equations": 0, "citations": 0}
    tier_a, tier_b, tier_c = 0, 0, 0
    for p in plans:
        res = compute_grounding_reward(p, gold, signal_set=("equations", "citations"))
        total["equations"] += res["per_signal"].get("equations", 0)
        total["citations"] += res["per_signal"].get("citations", 0)
        for em in res["diag"]["eq_matches"]:
            if em["tier"] == "a": tier_a += 1
            elif em["tier"] == "b": tier_b += 1
            elif em["tier"] == "c": tier_c += 1

    print(f"\nEquation hits across {len(plans)} plans: {total['equations']}")
    print(f"  tier (a) sympy strict: {tier_a}")
    print(f"  tier (b) sympy normalized: {tier_b}")
    print(f"  tier (c) fuzzy key_token: {tier_c}")
    print(f"Citation hits: {total['citations']}")
    print(f"Mean per-plan grounding score: {sum(compute_grounding_reward(p, gold)['score'] for p in plans) / len(plans):.4f}")
