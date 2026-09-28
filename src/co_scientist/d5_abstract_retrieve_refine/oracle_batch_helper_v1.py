"""D5 Phase 3 oracle batch helper — parse oracle_v2 medium.md → 181 items, split into rounds.

Phase 3 distillation pipeline (Path Y) reads `oracle_v2 medium.md` (181 typed
items across 6 categories: Insights / Methodology / Theory / Math / Empirical /
Failure_modes), shuffles with a fixed seed, and splits into 3 batches
(60 / 60 / 61) for the 3-round independent distillation pipeline.

**Source pool reframe (2026-04-27, Q3 lock revision)**:
Originally intended to use full oracle (`final.md`, 580 items, ~116K tokens),
but Qwen3-30B-A3B context window is 40K tokens — final does not fit even split
into 3 batches (each ~38K tokens, leaves no room for goal/output/critique).

Switched to `medium.md` (181 items, ~37K tokens, Opus relevance-filter ≥2):
- Same pool as σ baseline's slim.md (87 items, relevance ≥3) — slim ⊂ medium ⊂ final
- 3 batches × 60 items × ~200 tokens = ~12K tokens/batch, fits 40K with 24K margin
- Clean σ vs τ comparison: both pull from the same medium pool, only selection
  method differs (Opus hand-pick relevance≥3 vs 30B dynamic distillation)
- See `knowledge/current/RETRIEVAL_DESIGN_v1.md` for the full ML-scientist
  rationale (information theory + clean-experiment validity arguments)

This module is pure parsing + batching — no embeddings, no FAISS, no BM25
(Phase 3 walks Path Y not Path X). Items sent verbatim to the 30B model in
batches; the model selects + distills directly.

Source format expected:
    ## <Category>          ← H2 = section, e.g. "## Insights"
    ### <Category> N: <title>          ← H3 = item start
    - **Source**: ...
    - **Content**: ...
    - **Applicability to TTT-Discover goal**: ...
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

ORACLE_V2_DEFAULT_PATH = "projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/medium.md"

CATEGORIES = ("Insights", "Methodology", "Theory", "Math", "Empirical", "Failure_modes")

H2_RE = re.compile(r"^##\s+(\S+)\s*$")
H3_RE = re.compile(r"^###\s+(\S+)\s+(\d+):\s*(.*?)\s*$")


@dataclass(frozen=True)
class OracleItem:
    """One typed item from oracle_v2."""
    category: str           # e.g. "Insights"
    item_idx: int           # within-category index, e.g. 7 → "Insights 7"
    title: str              # heading title text
    body: str               # multi-line markdown body (Source + Content + Applicability)

    @property
    def label(self) -> str:
        """Stable id like 'Insights_007' for prompt rendering + dedup."""
        return f"{self.category}_{self.item_idx:03d}"

    def render(self) -> str:
        """Single string render suitable for prompt insertion."""
        return f"### {self.category} {self.item_idx}: {self.title}\n{self.body}"


def parse_oracle(path: str | Path) -> list[OracleItem]:
    """Parse oracle_v2 final.md into a flat list of OracleItem.

    Items appear in the original document order; downstream `make_batches`
    shuffles before splitting.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    items: list[OracleItem] = []
    current_category: str | None = None
    current_item: OracleItem | None = None
    body_accum: list[str] = []

    def flush_current():
        nonlocal current_item, body_accum
        if current_item is not None:
            items.append(OracleItem(
                category=current_item.category,
                item_idx=current_item.item_idx,
                title=current_item.title,
                body="\n".join(body_accum).strip(),
            ))
        current_item = None
        body_accum = []

    for line in lines:
        h2 = H2_RE.match(line)
        if h2:
            flush_current()
            cat = h2.group(1)
            current_category = cat if cat in CATEGORIES else None
            continue
        h3 = H3_RE.match(line)
        if h3 and current_category is not None and h3.group(1) == current_category:
            flush_current()
            current_item = OracleItem(
                category=current_category,
                item_idx=int(h3.group(2)),
                title=h3.group(3),
                body="",  # filled by flush_current()
            )
            continue
        if current_item is not None:
            body_accum.append(line)

    flush_current()
    return items


def make_batches(
    items: list[OracleItem],
    sizes: tuple[int, ...] = (60, 60, 61),
    seed: int = 42,
) -> list[list[OracleItem]]:
    """Shuffle items with the given seed, split into batches of the given sizes.

    Returns a list of batches; batches[i] is the i-th round's oracle context
    (independent across rounds per Q4.2 lock — model does NOT see prior batch
    distillations during distillation step).

    The default sizes (60, 60, 61) sum to 181 = full medium oracle_v2. If using
    a different source (slim 87 items, final 580 items), pass adjusted sizes.
    """
    if sum(sizes) > len(items):
        raise ValueError(f"sum(sizes)={sum(sizes)} > len(items)={len(items)}")
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    batches: list[list[OracleItem]] = []
    cursor = 0
    for size in sizes:
        batches.append(shuffled[cursor:cursor + size])
        cursor += size
    return batches


def render_batch(batch: list[OracleItem]) -> str:
    """Render a batch as a single string for prompt insertion."""
    return "\n\n".join(item.render() for item in batch)


def main() -> None:
    """Sanity test: parse oracle_v2, split into 3 batches, print stats."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / ORACLE_V2_DEFAULT_PATH
    items = parse_oracle(path)
    print(f"Parsed {len(items)} items from {path.name}")

    cat_counts = {cat: 0 for cat in CATEGORIES}
    for item in items:
        cat_counts[item.category] += 1
    print("Per-category:", cat_counts)

    batches = make_batches(items, sizes=(60, 60, 61), seed=42)
    print(f"Batches: {[len(b) for b in batches]} (total {sum(len(b) for b in batches)})")

    for i, batch in enumerate(batches, 1):
        rendered = render_batch(batch)
        print(f"  batch_{i}: {len(batch)} items, {len(rendered)} chars, ~{len(rendered)//4} tokens (rough)")

    sample = batches[0][0]
    print(f"\nSample item from batch_1[0]:")
    print(f"  label={sample.label}, title={sample.title!r}")
    print(f"  body[:160]={sample.body[:160]!r}")


if __name__ == "__main__":
    main()
