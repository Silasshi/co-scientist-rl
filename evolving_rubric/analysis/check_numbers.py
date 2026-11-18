"""Re-derive every number quoted in evolving_rubric/README.md from evolving_rubric/data/. Exits non-zero on mismatch."""
import json

import pandas as pd

from common.analysis import Checker
from evolving_rubric.analysis._load import D, items, pass_table

c = Checker()
for run, n, n_neg in [("smoke_no_rl", 7, 6), ("smoke_rl_coupled", 9, 6), ("dryrun_rl_coupled", 2, 1)]:
    its = items(run)
    c.check(f"{run}: rubric items generated", len(its), n, 0)
    c.check(f"{run}: of which negative (anti-Goodhart)", sum(i["rubric_type"] == "negative" for i in its), n_neg, 0)

t = pass_table("smoke_rl_coupled")
c.check("smoke_rl_coupled: rollouts graded per item per iteration", float(t.n_rollouts.max()), 8, 0)
c.check("smoke_rl_coupled: min per-item grade std at its last iteration", t.groupby("title")["std"].last().min(), 0.33, 5e-3)
m = pd.read_csv(D / "runs/smoke_rl_coupled/metrics.csv").set_index("iter")
c.check("logged R_active pass rate, iter 2", m.loc[2, "r_active/rollout_agg_mean"], 0.196, 5e-4)
c.check("logged R_active pass rate, iter 9", m.loc[9, "r_active/rollout_agg_mean"], 0.604, 5e-4)
allm = t.groupby("iter")["pass"].mean()
c.check("unweighted pass rate over all active items, iter 2", allm[2], 0.250, 5e-4)
c.check("unweighted pass rate over all active items, iter 9", allm[9], 0.569, 5e-4)
fixed = t[t.created <= 3].groupby("iter")["pass"].mean()
c.check("same 4 items (created <= iter 3), iter 3", fixed[3], 0.3125, 5e-4)
c.check("same 4 items (created <= iter 3), iter 9", fixed[9], 0.344, 5e-4)
pos = t[t.type == "positive"].groupby("iter")["pass"].mean()
c.check("positive items, mean pass rate iter 9", pos[9], 0.292, 5e-4)
neg_last = t[t.type == "negative"].groupby("title")["pass"].last()
c.check("negative items, mean pass rate at their last iteration", neg_last.mean(), 0.708, 5e-4)
anchor = [json.loads(line) for line in open(D / "runs/smoke_rl_coupled/rubric_anchor_log.jsonl")]
c.check("reference proposal pass rate on active items (iter 5)", anchor[0]["ref_aggregate"], 0.875)
c.report()
