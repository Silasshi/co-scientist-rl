"""Re-derive every number quoted in grpo/README.md from grpo/data/. Exits non-zero on mismatch."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.analysis import Checker, eval_mean_all, expected_max

D = Path(__file__).resolve().parents[1] / "data"
E = D / "eval"
c = Checker()

# --- binary (Goel-faithful) reward does not train -------------------------------------------
m = pd.read_csv(D / "train_metrics/binary_reward.csv")
c.check("binary: train reward batch 0", m["reward/total"].iloc[0], 0.1033)
c.check("binary: train reward final (batch 59)", m["reward/total"].iloc[-1], 0.0951)
t = pd.read_parquet(D / "train_samples/binary_reward.parquet")
g = t.groupby(["batch_idx", "group_idx"]).rubric_score
size, std = g.size(), g.std(ddof=0)
c.check("binary: GRPO groups with >=2 samples", float((size >= 2).sum()), 3554, 0)
c.check("binary: % of those groups with zero reward variance", 100 * float((std[size >= 2] == 0).mean()), 28.7, 0.05)
c.check("binary: % of samples scoring exactly 0", 100 * float((t.rubric_score == 0).mean()), 64.5, 0.05)

# --- graded reward + GRPO -----------------------------------------------------------------------
m = pd.read_csv(D / "train_metrics/grpo_main.csv")
c.check("main: train reward batch 0", m["reward/total"].iloc[0], 0.6394)
c.check("main: train reward final (batch 214)", m["reward/total"].iloc[-1], 0.7835)
base, n_gen, n_valid = eval_mean_all(E / "grpo_main.base_m1_summary.csv")
c.check("eval: base model (all 5120 samples)", base, 0.5601)
c.check("eval: base model valid share %", 100 * n_valid / n_gen, 76.2, 0.05)
trained, n_gen, n_valid = eval_mean_all(E / "grpo_main.eval_b214_summary.csv")
c.check("eval: GRPO b214 (all 5120 samples)", trained, 0.6926)
c.check("eval: relative gain over base %", 100 * (trained / base - 1), 23.7, 0.05)
c.check("eval: GRPO b214 valid share %", 100 * n_valid / n_gen, 100.0, 0.05)
for name, tag, val in [("arxiv", "grpo_arxiv.eval_b101", 0.6889), ("pubmed", "grpo_pubmed.eval_b99", 0.8263)]:
    c.check(f"eval: GRPO {name} split (trained)", eval_mean_all(E / f"{tag}_summary.csv")[0], val)
c.check("eval: GPT-5.4 zero-shot (640 goals x 1)", pd.read_parquet(E / "gpt54_zeroshot.sota.parquet").rubric_score.mean(), 0.8263)
c.check("eval: reference plans (n=685)", pd.read_parquet(E / "grpo_main.reference.parquet").rubric_score.mean(), 0.8665)

# --- selection gap (best-of-N on the trained checkpoint) ----------------------------------------
b = pd.read_parquet(E / "grpo_main.bon_b214.parquet")
goals = b.groupby(["eval_batch_idx", "goal_idx"])
for k, val in [(1, 0.6811), (2, 0.7535), (4, 0.8075), (8, 0.8494)]:
    c.check(f"BoN-{k}", goals.grader_rubric_score.apply(lambda s, k=k: expected_max(s.values, k)).mean(), val)
c.check("BoN self-selected", b.loc[b.is_self_pick == 1, "grader_rubric_score"].mean(), 0.6936)
rc = [spearmanr(d.grader_rubric_score, d.self_eval_score).correlation for _, d in goals]
c.check("self-eval vs grader rank corr (mean over goals)", float(np.nanmean(rc)), 0.0634)
share = (0.8494 - 0.6811) / (0.8665 - 0.6811)
c.check("share of mean->reference gap below oracle BoN-8 %", 100 * share, 90.8, 0.05)

# --- what else training changed: length shaping and output format -------------------------------
s = pd.read_csv(D / "train_metrics/grpo_main.batch_summary.csv")
c.check("main: mean plan length, first 10 batches (words)", s.length_mean.head(10).mean(), 405.6, 0.05)
c.check("main: mean plan length, last 10 batches (words)", s.length_mean.tail(10).mean(), 578.8, 0.05)
c.check("main: peak of 5-batch mean length (words)", s.length_mean.rolling(5, min_periods=1).mean().max(), 740, 5)
c.check("main: mean format penalty, first 10 batches", s.format_rate.head(10).mean(), 0.0479)
c.check("main: mean format penalty, last 10 batches", s.format_rate.tail(10).mean(), 0.0024)
for tag, val in [("grpo_main.base_m1", 0.5666), ("grpo_main.eval_b214", 0.6926)]:
    e = pd.read_parquet(E / f"{tag}.parquet")
    c.check(f"eval: {tag} valid-only mean", e.rubric_score.mean(), val)
eb, et = pd.read_parquet(E / "grpo_main.base_m1.parquet"), pd.read_parquet(E / "grpo_main.eval_b214.parquet")
bins = [0, 400, 500, 600, 700, 10_000]
gb = eb.groupby(pd.cut(eb.word_count, bins), observed=True).rubric_score.agg(["mean", "size"])
gt = et.groupby(pd.cut(et.word_count, bins), observed=True).rubric_score.agg(["mean", "size"])
# the eval gain holds inside every length bin, so it is not explained by the length change
j = gb.join(gt, lsuffix="_b", rsuffix="_t")
c.check("eval: length bins with a gain (of 5)", float((j.mean_t > j.mean_b).sum()), 5, 0)
c.check("eval: smallest within-bin gain (trained - base)", float((j.mean_t - j.mean_b).min()), 0.084, 5e-4)

# --- rubric score vs independent Opus depth audit -----------------------------------------------
a = pd.read_csv(D / "rubric_audit.csv")
sg, mg = a[a.audit_set == "same_goal"], a[a.audit_set == "mixed_goals"]
c.check("audit: same goal, n", float(len(sg)), 8, 0)
c.check("audit: same goal, Pearson(rubric, Opus)", sg.rubric_score.corr(sg.opus_total), 0.577)
c.check("audit: same goal, Spearman(rubric, Opus)", spearmanr(sg.rubric_score, sg.opus_total).correlation, 0.479)
c.check("audit: same goal, rubric min", sg.rubric_score.min(), 0.04)
c.check("audit: same goal, rubric max", sg.rubric_score.max(), 0.98, 0.005)
c.check("audit: same goal, Opus min", sg.opus_total.min(), 7, 0)
c.check("audit: same goal, Opus max", sg.opus_total.max(), 11, 0)
c.check("audit: mixed goals, Pearson(rubric, Opus)", mg.rubric_score.corr(mg.opus_total), -0.167)
top = mg.loc[mg.rubric_score.idxmax()]
c.check("audit: mixed goals, top rubric score", top.rubric_score, 0.9486)
c.check("audit: mixed goals, Opus score of that plan", top.opus_total, 7, 0)
c.check("audit: Opus min over all 13 plans", a.opus_total.min(), 6, 0)
c.check("audit: Opus max over all 13 plans", a.opus_total.max(), 11, 0)
c.report()
