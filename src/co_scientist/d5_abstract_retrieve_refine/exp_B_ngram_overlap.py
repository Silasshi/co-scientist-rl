"""Exp B — n-gram overlap trajectory across 6 v8-d5sdpo cells.

Falsifier for F16 H16-2 (critique-conditioned advantage rewards
critique-conformity, not oracle-fidelity). For each cell+iter, compute 4-gram
Jaccard overlap of each plan's solution body vs the slim oracle's
Math+Methodology corpus (12+17 = 29 items, hard-asserted).

Decision rule (pre-registered in EXPERIMENT_PLAN_oracle_transfer_ABC.md L100-105):
- All 6 cells flat or decreasing → H16-2 corroborated
- All 6 cells monotonic rise → H16-2 falsified
- Mixed → ambiguous; report case-by-case

Two panels: pre-cliff only vs full trajectory (cliff iter detected from
audit_responses/iter_*.json mean-total argmax-drop). Reference plan overlap
plotted as horizontal dashed anchor line for calibration.

Usage:
    PYTHONPATH=src python -m co_scientist.d5_abstract_retrieve_refine.exp_B_ngram_overlap
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import chz
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v1 import extract_solution


CELL_LABELS = {
    "base": "base (mask=F α=0)",
    "G": "G (mask=T α=0.05 PPO)",
    "Gplus": "G+ (mask=T α=0.1 PPO)",
    "E": "E (mask=F α=0.05 PPO)",
    "Cplus": "C+ (mask=T α=0.1 IS)",
    "Fplus": "F+ (mask=T α=0.05 IS)",
}


@chz.chz
class Config:
    runs_root: str = "projects/d5_abstract_retrieve_refine/runs"
    cell_dir_template: str = "2026_04_29_mu_v8_d5sdpo_{cell}"
    cells: tuple[str, ...] = ("base", "G", "Gplus", "E", "Cplus", "Fplus")
    oracle_slim_path: str = (
        "projects/d5_abstract_retrieve_refine/data/oracles/"
        "oracle_v2_2026_04_26_build/slim.md"
    )
    reference_plan_path: str = (
        "projects/d5_abstract_retrieve_refine/dataset/reference_solution.txt"
    )
    log_path: str = "projects/d5_abstract_retrieve_refine/runs/2026_04_29_exp_B_ngram_overlap"
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_SECTION_RE = re.compile(r"^### (Math|Methodology) (\d+):", re.MULTILINE)


def extract_math_methodology_corpus(slim_md: str) -> tuple[str, int]:
    """Concat Math + Methodology section bodies; assert exactly 29 items."""
    matches = list(_SECTION_RE.finditer(slim_md))
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(slim_md)
        blocks.append(slim_md[start:end])
    assert len(blocks) == 29, (
        f"Expected 29 Math+Methodology items (12 Math + 17 Methodology), got {len(blocks)}"
    )
    return "\n\n".join(blocks), len(blocks)


def load_buffer_rows(buffer_path: Path) -> list[dict]:
    rows = []
    with open(buffer_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_cliff_iter(audit_dir: Path) -> int | None:
    """Return iter index where the largest consecutive mean-total drop occurs.

    Reads audit_responses/iter_???.json (NOT *.partial.json), computes mean
    judgment.total per iter; cliff_iter = i+1 where mean[i] - mean[i+1] is max.
    Returns None if <2 complete iters.
    """
    if not audit_dir.exists():
        return None
    iter_files = sorted(audit_dir.glob("iter_???.json"))
    if len(iter_files) < 2:
        return None
    means = []
    for f in iter_files:
        try:
            data = json.loads(f.read_text())
            judgments = data.get("judgments", [])
            totals = [j.get("total") for j in judgments if j.get("total") is not None]
            means.append(float(np.mean(totals)) if totals else float("nan"))
        except Exception:
            means.append(float("nan"))
    if len(means) < 2:
        return None
    drops = [
        (means[i] - means[i + 1]) if not (np.isnan(means[i]) or np.isnan(means[i + 1])) else -np.inf
        for i in range(len(means) - 1)
    ]
    if all(d == -np.inf for d in drops):
        return None
    cliff_idx = int(np.argmax(drops))
    return cliff_idx + 1  # cliff iter is the one AFTER the drop


def compute_overlap_per_iter(
    rows: list[dict], reference_ngrams: set, n: int
) -> dict[int, list[float]]:
    """Group plans by iter, compute per-plan n-gram Jaccard, return iter -> list."""
    by_iter: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        plan_text = r.get("plan_text", "") or ""
        if not plan_text:
            continue
        tokens = _tokenize(plan_text)
        ng = _ngrams(tokens, n)
        by_iter[int(r["iter"])].append(jaccard(ng, reference_ngrams))
    return dict(by_iter)


def _slope_verdict(pre_iters: list[int], pre_means: list[float]) -> tuple[float, str]:
    if len(pre_iters) < 2:
        return float("nan"), "INSUFFICIENT"
    slope = float(np.polyfit(pre_iters, pre_means, 1)[0])
    if slope > 1e-4:
        return slope, "RISING"
    if slope < -1e-4:
        return slope, "FALLING"
    return slope, "FLAT"


def main(config: Config) -> None:
    log_dir = _resolve(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = log_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ----- Reference corpus (Math + Methodology from slim) -----
    slim_md = _resolve(config.oracle_slim_path).read_text()
    corpus_text, n_items = extract_math_methodology_corpus(slim_md)
    corpus_tokens = _tokenize(corpus_text)
    print(
        f"[ok] Math+Methodology corpus: {n_items} items, "
        f"{len(corpus_text)} chars, {len(corpus_tokens)} tokens"
    )

    # ----- Reference plan anchor (preload, tokenize once) -----
    ref_text = extract_solution(_resolve(config.reference_plan_path).read_text())
    ref_tokens = _tokenize(ref_text)
    print(f"[ok] reference plan: {len(ref_tokens)} tokens")

    # ----- Preload all cell rows once (heavy file reads) -----
    cell_rows: dict[str, list[dict]] = {}
    cell_cliff: dict[str, int | None] = {}
    for cell in config.cells:
        cell_dir = _resolve(config.runs_root) / config.cell_dir_template.format(cell=cell)
        buffer_path = cell_dir / "buffer.jsonl"
        if not buffer_path.exists():
            print(f"[warn] missing {buffer_path} — skipping")
            continue
        cell_rows[cell] = load_buffer_rows(buffer_path)
        cell_cliff[cell] = detect_cliff_iter(cell_dir / "audit_responses")

    # ----- Sweep n-gram orders -----
    csv_lines = ["cell,n,iter,n_plans,mean_overlap,std_overlap,is_post_cliff"]
    all_verdicts: dict[int, dict] = {}

    n_orders = list(config.ngram_orders)
    fig, axes = plt.subplots(len(n_orders), 2, figsize=(14, 4 * len(n_orders)))
    if len(n_orders) == 1:
        axes = np.array([axes])

    for row, n in enumerate(n_orders):
        corpus_ngrams = _ngrams(corpus_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        ref_overlap = jaccard(ref_ngrams, corpus_ngrams)
        print(
            f"\n[n={n}] corpus unique={len(corpus_ngrams)}, "
            f"reference plan overlap={ref_overlap:.4f}"
        )

        cell_summary: dict[str, dict] = {}
        for cell in config.cells:
            if cell not in cell_rows:
                continue
            by_iter = compute_overlap_per_iter(cell_rows[cell], corpus_ngrams, n)
            if not by_iter:
                continue
            cliff = cell_cliff.get(cell)
            iters = sorted(by_iter.keys())
            means = [float(np.mean(by_iter[i])) for i in iters]
            stds = [float(np.std(by_iter[i])) for i in iters]
            for i, m, s in zip(iters, means, stds):
                n_plans = len(by_iter[i])
                is_post = int(cliff is not None and i >= cliff)
                csv_lines.append(
                    f"{cell},{n},{i},{n_plans},{m:.6f},{s:.6f},{is_post}"
                )
            cell_summary[cell] = dict(iters=iters, means=means, stds=stds, cliff=cliff)

        # Plot
        ax_pre, ax_full = axes[row][0], axes[row][1]
        for cell, summary in cell_summary.items():
            iters = summary["iters"]
            means = summary["means"]
            cliff = summary["cliff"]
            label = CELL_LABELS.get(cell, cell)
            if cliff is not None:
                pre = [(i, m) for i, m in zip(iters, means) if i < cliff]
                if pre:
                    ax_pre.plot([p[0] for p in pre], [p[1] for p in pre], marker="o", label=label)
            else:
                ax_pre.plot(iters, means, marker="o", label=label)
            line = ax_full.plot(iters, means, marker="o", label=label)[0]
            if cliff is not None:
                ax_full.axvline(cliff, alpha=0.25, linestyle=":", color=line.get_color())

        for ax, title in zip(
            [ax_pre, ax_full],
            [f"n={n} pre-cliff", f"n={n} full (cliff = dotted)"],
        ):
            ax.axhline(
                ref_overlap,
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"ref plan ({ref_overlap:.4f})",
            )
            ax.set_xlabel("iter")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, loc="best")
        ax_pre.set_ylabel(f"{n}-gram Jaccard")

        # Per-cell slope verdicts at this n
        verdicts = {}
        for cell, summary in cell_summary.items():
            iters = summary["iters"]
            means = summary["means"]
            cliff = summary["cliff"]
            pre_iters = [i for i in iters if cliff is None or i < cliff]
            pre_means = [means[iters.index(i)] for i in pre_iters]
            slope, vd = _slope_verdict(pre_iters, pre_means)
            verdicts[cell] = dict(slope=slope, verdict=vd, peak=max(means), iter0=means[0])

        all_verdicts[n] = dict(ref_overlap=ref_overlap, per_cell=verdicts)
        n_rising = sum(1 for v in verdicts.values() if v["verdict"] == "RISING")
        n_total = len(verdicts)
        if n_rising == n_total:
            joint = f"FALSIFIED at n={n} (all {n_total} cells rising)"
        elif n_rising == 0:
            joint = f"CORROBORATED at n={n} (no cells rising)"
        else:
            joint = f"MIXED at n={n} ({n_rising}/{n_total} rising)"
        all_verdicts[n]["joint"] = joint
        print(f"  per-cell verdicts: {[(c, v['verdict']) for c, v in verdicts.items()]}")
        print(f"  joint: {joint}")

    fig.suptitle(
        "Exp B — n-gram overlap trajectories across {1,2,3,4}-gram orders\n"
        "(H16-2: critique-conditioned advantage rewards critique-conformity, not oracle-fidelity)"
    )
    fig.tight_layout()
    fig_path = figures_dir / "exp_B_overlap_trajectories.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\n[ok] saved figure: {fig_path}")

    csv_path = log_dir / "per_cell_per_iter.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"[ok] saved CSV: {csv_path}")

    # Save verdicts JSON
    verdict_path = log_dir / "verdict.json"
    verdict_path.write_text(json.dumps(all_verdicts, indent=2))
    print(f"[ok] saved verdict: {verdict_path}")

    print("\n=== Joint verdicts across n-gram orders ===")
    for n in n_orders:
        print(f"  n={n}: {all_verdicts[n]['joint']}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
