# co-scientist-rl

**RL post-training of a 30B LLM to write research plans.**

I reproduced the rubric-reward GRPO recipe of Goel et al. ([arXiv:2512.23707](https://arxiv.org/abs/2512.23707)) on Qwen3-30B-A3B. Besides GRPO, I also implemented evolving rubrics (DR Tulu), self-distillation (SDPO), and test-time training with an entropic objective and UCB-based buffer reuse (TTT-Discover).

Each method folder contains:
- the implementation
- the experiment data (numeric, in the repo)
- an analysis: figures, results and takeaways, with a script that re-derives every quoted number from the data

Author: Yuhong (Silas) Shi, Torr Vision Group, University of Oxford (Dec 2025 – Apr 2026).

| | Method | Follows | Main result | Takeaway |
|---|---|---|---|---|
| [**`grpo/`**](grpo/) | GRPO with an LLM-judge rubric reward | Goel et al. 2025; DeepSeekMath | Held-out rubric score **0.560 → 0.693** (640 goals × 8) | A binary rubric reward gives no gradient (29% of groups had zero variance), so a graded scale is required. The remaining gap is selection: best-of-8 reaches 0.849, but the model cannot pick it (ρ = 0.06) |
| [**`evolving_rubric/`**](evolving_rubric/) | Rubrics regenerated from current rollouts, variance-filtered, coupled into RL | DR Tulu (Shao et al. 2025) | Mechanism runs (≤ 10 iterations): 9 evolved items, 6 of them naming concrete reward-gaming behaviours | A rising rubric pass rate is mostly *composition* (0.20 → 0.60 overall, 0.31 → 0.34 on a fixed item set), so evaluate on a fixed item set. No training-effect claim |
| [**`sdpo/`**](sdpo/) | Self-distillation: the model conditioned on grader feedback is its own teacher | SDPO (Hübotter et al. 2026) | Ahead of a same-script GRPO control for 80/107 batches (peak 0.607 vs 0.591), then collapses; final eval 0.552 vs 0.607 | Self-distillation speeds up early learning but is unstable late; select checkpoints on validation |
| [**`ttt/`**](ttt/) | Per-goal test-time training: UCB buffer reuse, critique–revise, entropic weights | TTT-Discover (Yuksekgonul et al. 2026) | Critique conditioning **+0.19**; RL adds +0.09 at best-of-1 and nothing at best-of-2 (single runs) | Most of the gain comes from critique, not RL. Validate the grader first: the old signal set scored 0.975 while the independent judge gave 7–8 / 20 |

All training runs: Qwen3-30B-A3B (MoE), LoRA rank 64, trained through the [Tinker](https://thinkingmachines.ai/tinker/) API. Evaluation details and caveats are in each folder's README. Single runs and single-judge scores are flagged where they occur.

## Repository layout
```
grpo/  sdpo/  ttt/  evolving_rubric/     one folder per method:
    <code>.py                          implementation (importable package; python -m <method>.<module>)
    configs/ or data/runs/*/config.json   configs of the reported runs
    data/                              experiment data + data/README.md (protocol, what is excluded)
    analysis/make_figures.py           regenerates analysis/figures/*.png from data/
    analysis/check_numbers.py          re-derives every number in the README from data/
    tests/                             unit tests (no network)
common/                                shared: Tinker client profiles, hard-gate reward, grant reward, analysis helpers
READING.md                             the papers behind each method and how each was used
```

## Setup
```bash
pip install -r requirements.txt
export TINKER_API_KEY=...        # only needed to train; or CO_SCIENTIST_API_KEY_<NAME> + api_profile=<name>
```
Training needs a Tinker account. The dataset `facebook/research-plan-gen` is loaded from Hugging Face; the grant-proposal goals for `evolving_rubric/` must be supplied by you (see its `data/README.md`).

## Check the results without an account
```bash
pytest                                            # 151 unit tests
for m in grpo sdpo ttt evolving_rubric; do
  python -m $m.analysis.check_numbers             # every README number, recomputed from data/
  python -m $m.analysis.make_figures              # every figure, regenerated from data/
done
```

## History
This repository first held the full research project: six directions, 168 modules, decision logs and run summaries. It now focuses on the four methods above.
- The complete project is preserved on the [`archive`](https://github.com/Silasshi/co-scientist-rl/tree/archive) branch.
- The original snapshot is preserved at the [`v0-snapshot`](https://github.com/Silasshi/co-scientist-rl/tree/v0-snapshot) tag.

## Not included
Model checkpoints, generated plans and grader outputs (text), the dataset, paper drafts, and API credentials.

## License and attribution
Apache-2.0 ([`LICENSE`](LICENSE)). The training loop is adapted from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (Apache-2.0); see [`NOTICE`](NOTICE) and `THIRD_PARTY_NOTICES/`. The task and rubric-reward method follow Goel et al. (2025).
