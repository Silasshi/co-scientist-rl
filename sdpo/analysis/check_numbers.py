"""Re-derive every number quoted in sdpo/README.md from sdpo/data/. Exits non-zero on mismatch."""
from pathlib import Path

import pandas as pd

from common.analysis import Checker, eval_mean_all

D = Path(__file__).resolve().parents[1] / "data"
c = Checker()

for name, tag, all_v, valid_share, valid_mean in [
    ("sdpo", "eval_b106", 0.5517, 63.2, 0.5702),
    ("grpo_control", "eval_b106", 0.6066, 92.3, 0.6120),
    ("sdpo_warmstart", "eval_b235", 0.6823, 100.0, 0.6823),
]:
    mean, n_gen, n_valid = eval_mean_all(D / f"eval/{name}.{tag}_summary.csv")
    c.check(f"{name}: eval rubric, all 5120 samples", mean, all_v)
    c.check(f"{name}: eval valid share %", 100 * n_valid / n_gen, valid_share, 0.05)
    c.check(f"{name}: eval rubric, valid samples only", pd.read_parquet(D / f"eval/{name}.{tag}.parquet").rubric_score.mean(), valid_mean)

for name, first, last in [("sdpo", 0.464, 0.438), ("grpo_control", 0.465, 0.575)]:
    r = pd.read_csv(D / f"train_metrics/{name}.csv")["reward/total"]
    c.check(f"{name}: train reward, mean of first 10 batches", r.iloc[:10].mean(), first, 5e-4)
    c.check(f"{name}: train reward, mean of last 10 batches", r.iloc[-10:].mean(), last, 5e-4)

for name, last in [("sdpo", 26.4), ("grpo_control", 59.8)]:
    b = pd.read_csv(D / f"train_metrics/{name}.batch_summary.csv")
    c.check(f"{name}: format-noncompliant drops per batch, last 10", b["drops/format_noncompliant"].iloc[-10:].mean(), last, 0.05)

s = pd.read_csv(D / "train_metrics/sdpo.csv")
g = pd.read_csv(D / "train_metrics/grpo_control.csv")
rs, rg = s["reward/total"].rolling(5, min_periods=1).mean(), g["reward/total"].rolling(5, min_periods=1).mean()
c.check("sdpo: peak 5-batch train reward", rs.max(), 0.6069)
c.check("sdpo: batch of that peak", float(s["progress/batch"][rs.idxmax()]), 68, 0)
c.check("control: peak 5-batch train reward", rg.max(), 0.5912)
c.check("sdpo ahead of control (5-batch means), batches out of 107", float((rs > rg).sum()), 80, 0)
c.check("last batch where sdpo is ahead", float(s["progress/batch"][(rs > rg)[lambda x: x].index.max()]), 86, 0)
c.check("sdpo: train reward mean, batches 90-106", s["reward/total"].iloc[90:].mean(), 0.469, 5e-4)
c.check("control: train reward mean, batches 90-106", g["reward/total"].iloc[90:].mean(), 0.576, 5e-4)
bs = pd.read_csv(D / "train_metrics/sdpo.batch_summary.csv")
c.check("sdpo: SDPO-advantage std, batches 0-29", bs["advantage/sdpo_std"].iloc[:30].mean(), 0.725, 5e-4)
c.check("sdpo: SDPO-advantage std, batches 90-106", bs["advantage/sdpo_std"].iloc[90:].mean(), 0.404, 5e-4)
w = pd.read_csv(D / "train_metrics/sdpo_warmstart.csv")["reward/total"]
c.check("warmstart: train reward final batch", w.iloc[-1], 0.6734)
c.report()
