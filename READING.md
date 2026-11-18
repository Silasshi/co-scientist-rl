# Reading list

These are the papers behind this project, grouped by **how they were used**:
- **Built on** — the method is implemented or adapted in this repo.
- **Informed design** — it shaped a protocol or a decision.
- **Positioned against** — related work the D5 method was compared to.
- **Surveyed** — read for direction-setting (April 2026 literature synthesis) but not adopted.

I picked these by hand from a larger automated literature sweep. Every arXiv ID was checked against the paper's title.

## RL for LLMs — built on
| Paper | How it was used |
|---|---|
| Goel et al., *Training AI Co-Scientists Using Rubric Rewards* — [2512.23707](https://arxiv.org/abs/2512.23707) | The task, dataset and reward this project reproduces at 30B ([GRPO + rubric reward](docs/algorithms/01-grpo-rubric-reward.md)) |
| Shao et al., *DeepSeekMath* (GRPO) — [2402.03300](https://arxiv.org/abs/2402.03300) | Group-relative advantage, with no value network, in every trainer |
| Schulman et al., *Proximal Policy Optimization* — [1707.06347](https://arxiv.org/abs/1707.06347) | PPO-clip loss (ε = 0.2) used through Tinker `forward_backward` |
| Hu et al., *LoRA* — [2106.09685](https://arxiv.org/abs/2106.09685) | Rank-64 adapters on Qwen3-30B-A3B for all runs |
| Hübotter et al., *Reinforcement Learning via Self-Distillation* (SDPO) — [2601.20802](https://arxiv.org/abs/2601.20802) | Self-teacher conditioned on grader feedback; Δlog-prob as a dense advantage ([self-distillation](docs/algorithms/02-self-distillation.md)) |
| Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation* (OPSD) — [2601.18734](https://arxiv.org/abs/2601.18734) | Closest published counterpart to the on-policy OPD trainer; used to correct the naming of the in-house variants |
| Andrychowicz et al., *Hindsight Experience Replay* — [1707.01495](https://arxiv.org/abs/1707.01495) | Hindsight-relabelling idea behind the off-policy μ-v4 self-distillation variant |
| Yuksekgonul et al., *Learning to Discover at Test Time* (TTT-Discover) — [2601.16175](https://arxiv.org/abs/2601.16175) | Entropic objective with adaptive β, and per-problem test-time RL ([critique–revise](docs/algorithms/04-critique-revise-rl.md)) |
| *OpenClaw-RL: Train Any Agent Simply by Talking* — [2603.10165](https://arxiv.org/abs/2603.10165) | Process reward + OPD pipeline adapted for multi-turn plan refinement ([`experiments/rubric_reward/multiturn/train_multiturn_v2.py`](experiments/rubric_reward/multiturn/train_multiturn_v2.py)) |
| Chen et al., *Learning to Self-Evolve* (LSE) — [2603.18620](https://arxiv.org/abs/2603.18620) | Single-step improvement-reward RL for grant proposals ([`experiments/grant_proposal/train_lse_v1.py`](experiments/grant_proposal/train_lse_v1.py)); abandoned at +0.027 over the frozen model, within noise |

## Rewards and rubrics — built on
| Paper | How it was used |
|---|---|
| Gunjal et al., *Rubrics as Rewards* — [2507.17746](https://arxiv.org/abs/2507.17746) | "Decompose, don't collapse": keep rubric dimensions as separate signals. This is the argument behind the [vector advantage](docs/algorithms/03-multi-signal-vector-reward.md) |
| Wu et al., *Fine-Grained Human Feedback Gives Better Rewards* — [2306.01693](https://arxiv.org/abs/2306.01693) | Per-dimension dense rewards beat a single holistic score; part of the same consensus |
| Wang et al., *Interpretable Preferences via Multi-Objective Reward Modeling* (ArmoRM) — [2406.12845](https://arxiv.org/abs/2406.12845) | Interpretable multi-objective reward with explicit per-objective weights; part of the same consensus |
| *DR Tulu: RL with Evolving Rubrics for Deep Research* — [2511.19399](https://arxiv.org/abs/2511.19399) | Discriminative-power criteria in the rubric-generator prompt ([evolving rubrics](docs/algorithms/05-evolving-rubrics.md)) |
| *Online Rubrics Elicitation from Pairwise Comparisons* — [2510.07284](https://arxiv.org/abs/2510.07284) | Pairwise-contrast rubric elicitation (`select_contrast_pair`) |

## LLM-as-judge — informed design
| Paper | How it was used |
|---|---|
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench* — [2306.05685](https://arxiv.org/abs/2306.05685) | Position bias, hence hidden A/B swapping in pairwise judging ([judge protocols](docs/algorithms/06-llm-judge-evaluation.md)) |
| Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations* — [2404.13076](https://arxiv.org/abs/2404.13076) | Self-preference, hence a cross-family judge (Claude Opus) for every result after D1 |
| Verga et al., *Replacing Judges with Juries* (PoLL) — [2404.18796](https://arxiv.org/abs/2404.18796) | A panel of small judges from different model families beats one large judge, at lower cost; relevant to checking signals against several model families |
| Kim et al., *Prometheus 2* — [2405.01535](https://arxiv.org/abs/2405.01535) | Reference-based grading beats reference-free grading for most judges; the D1 grader is reference-anchored |

## Reward hacking — surveyed
| Paper | Relation |
|---|---|
| *CREAM: Consistency Regularized Self-Rewarding LMs* — [2410.12735](https://arxiv.org/abs/2410.12735) | Consistency regularisation across repeated self-gradings. Proposed for D3 ([`FOCUSED_PLAN_2026.md`](projects/ttt_discover/FOCUSED_PLAN_2026.md)); what shipped is repeated grading with a median and range-based confidence ([`train_buffer_ttt.py` L184](src/co_scientist/grant_proposal/train_buffer_ttt.py#L184)) |
| *MONA: Myopic Optimization with Non-myopic Approval* — [2501.13011](https://arxiv.org/abs/2501.13011) | Myopic optimisation as a defence against multi-step reward hacking; surveyed, not adopted |
| *Reward Shaping to Mitigate Reward Hacking in RLHF* (PAR) — [2502.18770](https://arxiv.org/abs/2502.18770) | Bounded reward shaping against reward hacking; surveyed, not adopted |

## Positioned against (D5 related work)
| Paper | Difference from D5 |
|---|---|
| *RLAD: Training LLMs to Discover Abstractions* — [2510.02263](https://arxiv.org/abs/2510.02263) | Abstraction + RL, but math-only, single round, no retrieval |
| Jiang et al., *Active Retrieval Augmented Generation* (FLARE) — [2305.06983](https://arxiv.org/abs/2305.06983) | Multi-round retrieval, but iterates per sentence, not per abstraction, and has no self-distillation |

## AI for science and agent harnesses — surveyed
| Paper | Takeaway recorded at the time |
|---|---|
| *Accelerating scientific discovery with Co-Scientist* (Google) — [2502.18864](https://arxiv.org/abs/2502.18864) | Multi-agent generate–debate–evolve system; a prompting-based reference point for the RL approach |
| *AI-Researcher: Autonomous Scientific Innovation* — [2505.18705](https://arxiv.org/abs/2505.18705) | End-to-end pipeline from literature review to paper |
| *DeepInnovator* — [2602.18920](https://arxiv.org/abs/2602.18920) | Process-oriented "delta" reward for ideation without ground truth; related to the revision-delta advantage in critique–revise |
| *Why LLMs Aren't Scientists Yet* — [2601.03315](https://arxiv.org/abs/2601.03315) | Failure analysis of four autonomous research attempts |
| *Aster* — [2602.07040](https://arxiv.org/abs/2602.07040) | Autonomous discovery agent with fast iteration |
| *K-Search* — [2602.19128](https://arxiv.org/abs/2602.19128) | LLM as a co-evolving world model with stagnation-bounded refinement |
| *AutoHarness* — [2603.03329](https://arxiv.org/abs/2603.03329) / *Meta-Harness* — [2603.28052](https://arxiv.org/abs/2603.28052) | Searching over the code that wraps the model instead of its weights; considered as an alternative to RL for D3 |
| *MetaClaw* — [2603.17187](https://arxiv.org/abs/2603.17187) | Dual-timescale adaptation: a fast skill library plus slow LoRA updates |
| *Enhancing LLM Planning through Intrinsic Self-Critique* — [2512.24103](https://arxiv.org/abs/2512.24103) | Self-critique works when the prompt carries the domain definition. A useful contrast to the rubric-blind self-selection here, which was near chance ([selection gap](docs/algorithms/07-selection-gap-bon.md)) |
