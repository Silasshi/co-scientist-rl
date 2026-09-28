"""D6 signal-validation dataset builder v1 — wash D4-v2 grader_panel_v8 into D6.

One-time script. Reads from:
  projects/grant_proposal_v2/analysis/grader_panel_v8/
    test_plans/plan_{00..14}.txt
    test_plans/sources.jsonl
    grades/opus.jsonl              (15 plans x 10 D4-v8 signals)
    grades/opus_depth_audit.jsonl  (15 plans, holistic 4-dim audit)
    grades/{model}.jsonl           (8 cross-grader baselines)

Writes to:
  projects/d6_grant_proposal/dataset/signal_validation/foundopt_opus/
    proposals/plan_{00..14}.md     (frontmatter + body)
    opus_grades.jsonl              (180 rows = 15 plans x 12 D6 signals; G3 + G5 null)
    opus_depth.jsonl               (15 holistic-audit rows, passthrough)
    metadata.jsonl                 (per-plan summary)
    panel_baselines/{model}.jsonl  (8 cross-grader files, passthrough)
    README.md                      (dataset card)

Goal: provide an Opus-anchored, quality-stratified dataset for Phase 2 signal
validation (qwen30b_vs_opus correlation under the production GRPO grader path).

Coverage caveat: D4-v8 redesign removed G3 (Technical Evidence) and G5 (Gap
Identification). Opus did NOT score these in this dataset; they're emitted as
opus_score=null with explanatory reasoning.

Usage:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.build_validation_dataset
    # or with overrides:
    PYTHONPATH=src python -m co_scientist.d6_grant_proposal.build_validation_dataset \
        force_rebuild=True
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import chz

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.d6_grant_proposal.signals import SIGNALS  # canonical 12

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]

# Signals removed in the D4-v8 redesign — Opus did not score these in grader_panel_v8.
ABSENT_IN_V8 = {"G3_technical_evidence", "G5_gap_identification"}
ABSENT_REASON = "not scored in grader_panel_v8 — signal removed during D4-v8 redesign"

PANEL_BASELINE_FILES = [
    "qwen3_235b.jsonl",
    "qwen3_30b.jsonl",
    "qwen3_4b.jsonl",
    "gpt_oss_120b.jsonl",
    "gpt_oss_20b.jsonl",
    "deepseek_v3_1.jsonl",
    "llama_3_1_8b.jsonl",
    "kimi_k2_thinking.jsonl",
]


@chz.chz
class Config:
    source_dir: str = "projects/grant_proposal_v2/analysis/grader_panel_v8"
    target_dir: str = "projects/d6_grant_proposal/dataset/signal_validation/foundopt_opus"
    goal_id: str = "ai/01_foundopt"
    force_rebuild: bool = False


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def _frontmatter(meta: dict, goal_id: str, char_count: int) -> str:
    lines = [
        "---",
        f"plan_id: {meta['plan_id']}",
        f"goal_id: {goal_id}",
        f"quality_bucket: {meta['quality_bucket']}",
        f"source_run: {meta['source_run']}",
    ]
    if meta.get("iteration") is not None:
        lines.append(f"iteration: {meta['iteration']}")
    if meta.get("entry_type"):
        lines.append(f"entry_type: {meta['entry_type']}")
    if meta.get("aggregate_reward") is not None:
        lines.append(f"qwen30b_aggregate_reward: {meta['aggregate_reward']}")
    lines += [
        f"word_count: {meta['word_count']}",
        f"char_count: {char_count}",
        "---",
        "",
    ]
    return "\n".join(lines)


def main(config: Config) -> None:
    source = (REPO_ROOT / config.source_dir).resolve()
    target = (REPO_ROOT / config.target_dir).resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if target.exists() and not config.force_rebuild:
        existing = list(target.iterdir())
        if existing:
            logger.warning(
                "Target already exists with %d entries: %s. Pass force_rebuild=True to rebuild.",
                len(existing), target,
            )
            return

    target.mkdir(parents=True, exist_ok=True)
    (target / "proposals").mkdir(exist_ok=True)
    (target / "panel_baselines").mkdir(exist_ok=True)

    # ---- 1. Load sources.jsonl ----
    sources = _read_jsonl(source / "test_plans" / "sources.jsonl")
    sources_by_id = {row["plan_id"]: row for row in sources}
    logger.info("Loaded %d source rows", len(sources))

    # ---- 2. Write proposals/plan_NN.md ----
    metadata_rows: list[dict] = []
    for meta in sources:
        plan_id = meta["plan_id"]
        body = (source / "test_plans" / f"{plan_id}.txt").read_text()
        char_count = len(body)
        word_count_actual = len(body.split())

        front = _frontmatter(meta, config.goal_id, char_count)
        (target / "proposals" / f"{plan_id}.md").write_text(front + body)

        metadata_rows.append({
            "plan_id": plan_id,
            "goal_id": config.goal_id,
            "quality_bucket": meta["quality_bucket"],
            "source_run": meta["source_run"],
            "iteration": meta.get("iteration"),
            "entry_type": meta.get("entry_type"),
            "qwen30b_aggregate_reward": meta.get("aggregate_reward"),
            "word_count_sources": meta["word_count"],
            "word_count_actual": word_count_actual,
            "char_count": char_count,
        })
    logger.info("Wrote %d proposals", len(metadata_rows))

    # ---- 3. Build opus_grades.jsonl (180 rows = 15 plans x 12 signals) ----
    raw_opus = _read_jsonl(source / "grades" / "opus.jsonl")
    opus_lookup = {(r["plan_id"], r["signal_id"]): r for r in raw_opus}

    canonical_signal_ids = [s.id for s in SIGNALS]
    assert len(canonical_signal_ids) == 12, f"expected 12 canonical signals, got {len(canonical_signal_ids)}"

    grades_rows: list[dict] = []
    plan_ids_sorted = sorted(sources_by_id.keys())
    for plan_id in plan_ids_sorted:
        for signal_id in canonical_signal_ids:
            if signal_id in ABSENT_IN_V8:
                grades_rows.append({
                    "plan_id": plan_id,
                    "signal_id": signal_id,
                    "opus_score": None,
                    "reasoning": ABSENT_REASON,
                    "critique": None,
                    "scored_in_v8": False,
                })
            else:
                src_row = opus_lookup.get((plan_id, signal_id))
                if src_row is None:
                    raise RuntimeError(
                        f"Missing Opus grade for ({plan_id}, {signal_id}) — corrupt source data"
                    )
                grades_rows.append({
                    "plan_id": plan_id,
                    "signal_id": signal_id,
                    "opus_score": src_row["score"],
                    "reasoning": src_row.get("reasoning"),
                    "critique": src_row.get("critique"),
                    "scored_in_v8": True,
                })
    _write_jsonl(target / "opus_grades.jsonl", grades_rows)
    logger.info("Wrote opus_grades.jsonl: %d rows (15 plans x 12 signals)", len(grades_rows))
    assert len(grades_rows) == 15 * 12, "row count mismatch"

    # ---- 4. Passthrough opus_depth.jsonl ----
    depth_src = source / "grades" / "opus_depth_audit.jsonl"
    shutil.copyfile(depth_src, target / "opus_depth.jsonl")
    depth_rows = _read_jsonl(target / "opus_depth.jsonl")
    logger.info("Copied opus_depth.jsonl: %d rows", len(depth_rows))
    assert len(depth_rows) == 15

    # ---- 5. Write metadata.jsonl ----
    _write_jsonl(target / "metadata.jsonl", metadata_rows)
    logger.info("Wrote metadata.jsonl: %d rows", len(metadata_rows))

    # ---- 6. Copy panel baselines ----
    copied = []
    for fname in PANEL_BASELINE_FILES:
        src_path = source / "grades" / fname
        if not src_path.exists():
            logger.warning("Panel baseline missing: %s — skipping", src_path)
            continue
        shutil.copyfile(src_path, target / "panel_baselines" / fname)
        copied.append(fname)
    logger.info("Copied %d panel-baseline files: %s", len(copied), copied)

    # ---- 7. Write README dataset card ----
    (target / "README.md").write_text(_dataset_card(len(grades_rows), copied))
    logger.info("Wrote README.md (dataset card)")

    logger.info("Done. Dataset at: %s", target)


def _dataset_card(n_grade_rows: int, panel_files: list[str]) -> str:
    return f"""# D6 Signal Validation Dataset — v8 Opus FoundOpt

**Source:** `projects/grant_proposal_v2/analysis/grader_panel_v8/` (D4-v2)
**Goal:** `ai/01_foundopt` (FoundOpt — single-domain, AI/CS optimisation)
**Built:** 2026-04-30 by `build_validation_dataset.py`
**Purpose:** Phase 2 of the D6 roadmap — quantify how well the Qwen3-30B-A3B production GRPO grader correlates with Opus-4.7 ground truth on stratified-quality grant proposals.

## Provenance

15 grant proposals on the FoundOpt goal, sampled by `grader_panel_v8/scripts/sample_test_plans.py` (seed=42) on 2026-04-22:

| Bucket | n | Source |
|---|---|---|
| `reference_anchor` | 3 | `reference_proposal.md` duplicated 3× (grader noise floor) |
| `high` | 4 | best plans (highest source-run Qwen-30B aggregate) from 4 different methods |
| `mid` | 4 | median plans (middle iters) from 4 different methods |
| `low` | 4 | early-iter (iter 0–2) fresh plans, no critique-revise |

**Quality bucket is determined by Qwen3-30B aggregate at gen time, NOT by Opus.** Use `opus_score` (this dataset) for quality ranking; the `quality_bucket` field reflects how the proposal was sampled, not the ground truth.

Each plan was scored independently by Opus 4.7 on **10 of D6's 12 signals** (1–5 scale + reasoning + critique). Opus also produced a holistic 4-dimension depth audit per plan (1–10 per dimension; rubric-agnostic).

8 additional graders (Qwen3-235B / 30B / 4B, GPT-OSS-120B / 20B, DeepSeek-V3.1, Llama-3.1-8B, Kimi-K2-Thinking) graded the same plans for cross-grader comparison.

## Files

```
foundopt_opus/
├── README.md                       # this file
├── proposals/
│   └── plan_{{00..14}}.md          # YAML frontmatter + body
├── opus_grades.jsonl               # {n_grade_rows} rows (15 plans × 12 D6 signals)
├── opus_depth.jsonl                # 15 rows (holistic 4-dim audit per plan)
├── metadata.jsonl                  # 15 rows (per-plan summary)
└── panel_baselines/
    └── {{model}}.jsonl             # {len(panel_files)} cross-grader files (passthrough)
```

## Schemas

### `proposals/plan_NN.md`

```markdown
---
plan_id: plan_NN
goal_id: ai/01_foundopt
quality_bucket: reference_anchor | high | mid | low
source_run: <run name or "reference">
iteration: <int or absent for reference>
entry_type: <e.g. "fresh", "revised", "reference">
qwen30b_aggregate_reward: <0.0–1.0 or absent for reference>
word_count: <from sources.jsonl>
char_count: <recomputed>
---

<proposal body — full grant proposal text>
```

### `opus_grades.jsonl`

```json
{{"plan_id": "plan_00", "signal_id": "G1_problem_specificity",
 "opus_score": 4, "reasoning": "...", "critique": "...",
 "scored_in_v8": true}}
```

For G3 (Technical Evidence) and G5 (Gap Identification) — **removed from the D4 rubric in the v8 redesign** — `opus_score` is `null`, `reasoning` is the explanatory string `"{ABSENT_REASON}"`, `critique` is `null`, and `scored_in_v8` is `false`. To re-enable G3 + G5 you'd need a separate Opus pass on the 15 proposals (≈ 30 calls).

### `opus_depth.jsonl`

```json
{{"plan_id": "plan_00", "model": "opus_4_7_depth_audit",
 "depth": 8, "methods": 8, "feasibility": 8, "grounding": 9, "total": 33}}
```

4 dimensions ∈ [1, 10]; total ∈ [4, 40]. **Rubric-agnostic** — Opus was not given the 12-signal scaffolding, so this is the strongest available "ground truth aggregate" for Phase 2's depth-audit correlation test.

### `metadata.jsonl`

Per-plan `{{plan_id, goal_id, quality_bucket, source_run, iteration, entry_type, qwen30b_aggregate_reward, word_count_sources, word_count_actual, char_count}}`. `word_count_actual` is recomputed from the body; `word_count_sources` is the value carried from `sources.jsonl`.

### `panel_baselines/<model>.jsonl`

Same schema as `opus.jsonl` (input, before re-shaping for D6) — `{{plan_id, signal_id, model, score, reasoning, ...}}`. Only the 10 v8-active signals are scored. Useful as cross-grader validation in Phase 2.

## Cross-grader benchmarks (from `grader_panel_v8/FINAL_SUMMARY.md`, 2026-04-22)

Mean per-signal Spearman ρ vs Opus on this same dataset (n=15):

| Grader | Mean ρ | Pooled ρ | MAE | Within-1 | Note |
|---|---:|---:|---:|---:|---|
| Qwen3-235B | **+0.741** | +0.774 | 0.63 | 87% | strong, 5.7× cost vs 30B |
| GPT-OSS-120B | +0.732 | +0.657 | 0.75 | 79% | cross-family check |
| GPT-OSS-20B | +0.674 | +0.659 | 0.71 | 80% | best small grader |
| DeepSeek-V3.1 | +0.563 | +0.690 | 0.66 | 81% | |
| **Qwen3-30B-A3B** | **+0.399** | +0.448 | 0.70 | 84% | **D6 production GRPO grader** |
| Qwen3-4B | +0.385 | +0.500 | 0.91 | 74% | |
| Llama-3.1-8B | +0.140 | +0.084 | 1.50 | 51% | unusable |

**Load-bearing for Phase 2:** Qwen3-30B (the actual GRPO reward signal) correlates with Opus at ρ ≈ 0.40 on this dataset. The D6 paper's "imperfect signal → GRPO ceiling" thesis stands or falls on confirming this number under the production GRPO scoring path (not a panel-style helper). Phase 2's `validate_signals.py` runs that test.

## Caveats

1. **n = 15.** Spearman ρ standard error ≈ 0.22. Differences between adjacently-ranked graders are within noise.
2. **Single goal (FoundOpt only).** Generalisation to `02_foundational_rl`, `08_ecosystem_dynamics`, or `12_climate_displacement` is not tested here. If a Phase 5 cross-goal claim is made, the grader-vs-Opus correlation must be re-measured per goal.
3. **Reference-anchor degeneracy.** `plan_00`, `plan_01`, `plan_02` are identical (the reference proposal copied 3×). Spearman is undefined on a degenerate triple — use them only for grader-noise estimates, not ranking.
4. **Quality buckets reflect Qwen-30B at gen time, not Opus.** Don't use `quality_bucket` as a quality ground truth; use `opus_score` or `opus_depth.total` instead.
5. **G3 / G5 absent.** 10/12 signal coverage — if D6's paper claim depends on G3 or G5 specifically, those need a separate Opus pass.
6. **Kimi-K2-Thinking partial.** Only 13/150 grades collected (latency); not representative.

## Reproduce

```bash
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.build_validation_dataset
# Force overwrite:
PYTHONPATH=src python -m co_scientist.d6_grant_proposal.build_validation_dataset force_rebuild=True
```

Source data is read-only; this script writes only into `dataset/signal_validation/foundopt_opus/`.
"""


if __name__ == "__main__":
    chz.entrypoint(main)
