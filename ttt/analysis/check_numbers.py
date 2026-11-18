"""Re-derive every number quoted in ttt/README.md from ttt/data/. Exits non-zero on mismatch."""
from pathlib import Path

import pandas as pd

from common.analysis import Checker

D = Path(__file__).resolve().parents[1] / "data"
c = Checker()
run = lambda n: pd.read_csv(D / f"runs/{n}/metrics.csv")  # noqa: E731
bmax = {n: run(n)["reward/buffer_max"].iloc[-1] for n in
        ["MAIN_v9", "B4_v9", "B4_stripped", "MAIN_fresh_bon1", "B4_bon1", "MAIN_fresh_bon2", "MAIN_v8.1", "B4_v8.1"]}
for n, v in [("MAIN_v9", 0.95), ("B4_v9", 0.95), ("B4_stripped", 0.7625), ("MAIN_fresh_bon1", 0.9275),
             ("B4_bon1", 0.84), ("MAIN_fresh_bon2", 0.925), ("MAIN_v8.1", 0.975), ("B4_v8.1", 0.975)]:
    c.check(f"best plan after 25 iters (training grader): {n}", bmax[n], v)
c.check("critique conditioning effect: B4_v9 - B4_stripped", bmax["B4_v9"] - bmax["B4_stripped"], 0.1875)
c.check("RL-on-fresh effect at best-of-1: MAIN_fresh_bon1 - B4_bon1", bmax["MAIN_fresh_bon1"] - bmax["B4_bon1"], 0.0875)
a, b = run("MAIN_v9")["reward/buffer_max"], run("B4_v9")["reward/buffer_max"]
ahead = (a > b)[lambda x: x].index
c.check("first iter where MAIN_v9 leads B4_v9", float(run("MAIN_v9")["iter"][ahead.min()]), 16, 0)
c.check("last iter where MAIN_v9 leads B4_v9", float(run("MAIN_v9")["iter"][ahead.max()]), 21, 0)
for n, v in [("MAIN_fresh_bon1", 0.433), ("B4_bon1", 0.425)]:
    c.check(f"mean fresh-plan reward over 25 iters: {n}", run(n)["fresh/reward_mean"].mean(), v, 5e-4)
c.check("adaptive beta at last iter (MAIN_fresh_bon1)", run("MAIN_fresh_bon1")["beta"].iloc[-1], 8.10, 5e-3)
o = pd.read_csv(D / "opus_judge_scores.csv").set_index("name")["opus_total_20"]
for n, v in [("MAIN_fresh_bon1", 10), ("B4_bon1", 8), ("MAIN_v9", 9), ("B4_v9", 9), ("B4_stripped", 6),
             ("MAIN_v8.1", 7), ("B4_v8.1", 8), ("reference", 20)]:
    c.check(f"Opus /20: {n}", float(o[n]), v, 0)
c.report()
