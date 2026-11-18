# Reading list

These are the papers behind the four methods in this repo, grouped by method. Each entry says how the paper was used. Every arXiv ID was checked against the paper's title.

## GRPO with a rubric reward — [`grpo/`](grpo/)
| Paper | How it was used |
|---|---|
| Goel et al., *Training AI Co-Scientists Using Rubric Rewards* — [2512.23707](https://arxiv.org/abs/2512.23707) | The task, dataset and rubric reward that this repo reproduces at 30B |
| Shao et al., *DeepSeekMath* (GRPO) — [2402.03300](https://arxiv.org/abs/2402.03300) | Group-relative advantage without a value network |
| Schulman et al., *Proximal Policy Optimization* — [1707.06347](https://arxiv.org/abs/1707.06347) | The PPO-clip loss |
| Hu et al., *LoRA* — [2106.09685](https://arxiv.org/abs/2106.09685) | Rank-64 adapters on Qwen3-30B-A3B for all runs |
| Gunjal et al., *Rubrics as Rewards* — [2507.17746](https://arxiv.org/abs/2507.17746) | Rubric-based rewards for non-verifiable tasks |
| Song et al., *Mind the Gap: Examining the Self-Improvement Capabilities of LLMs* — [2412.02674](https://arxiv.org/abs/2412.02674) | The generation–verification gap, which is the frame for the best-of-N selection-gap analysis |

## Self-distillation — [`sdpo/`](sdpo/)
| Paper | How it was used |
|---|---|
| Hübotter et al., *Reinforcement Learning via Self-Distillation* (SDPO) — [2601.20802](https://arxiv.org/abs/2601.20802) | The method implemented in `sdpo/train.py`: the self-teacher conditioned on feedback, mixed with the RL advantage, and regularised toward the initial policy |
| Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation* (OPSD) — [2601.18734](https://arxiv.org/abs/2601.18734) | The closest published counterpart to the `external_critic/` variant (on-policy, self-teacher, privileged context) |
| Agarwal et al., *On-Policy Distillation of Language Models* — [2306.13649](https://arxiv.org/abs/2306.13649) | On-policy distillation from a *separate* teacher. This repo does not do this; it is listed to make the distinction from self-distillation explicit |

## Test-time training — [`ttt/`](ttt/)
| Paper | How it was used |
|---|---|
| Yuksekgonul et al., *Learning to Discover at Test Time* (TTT-Discover) — [2601.16175](https://arxiv.org/abs/2601.16175) | Per-problem test-time RL, the entropic objective with adaptive β, and buffer-based reuse of past attempts (UCB in this repo, PUCT in the paper) |
| Zhang et al., *Critique-GRPO* — [2506.03106](https://arxiv.org/abs/2506.03106) | Learning from critique-guided refinements together with numerical rewards. Related to the critique–revise loop, whose effect turned out to dominate |

## Evolving rubrics — [`evolving_rubric/`](evolving_rubric/)
| Paper | How it was used |
|---|---|
| Shao et al., *DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research* (RLER) — [2511.19399](https://arxiv.org/abs/2511.19399) | Rubrics regenerated from current rollouts and filtered by grade variance; cited in `rubric_gen_prompt.py` |
| Rezaei et al., *Online Rubrics Elicitation from Pairwise Comparisons* — [2510.07284](https://arxiv.org/abs/2510.07284) | Pairwise-contrast elicitation of new rubric items (`select_contrast_pair`) |

## LLM-as-judge (independent evaluation in `ttt/`)
| Paper | How it was used |
|---|---|
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — [2306.05685](https://arxiv.org/abs/2306.05685) | Position bias and other judge biases |
| Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations* — [2404.13076](https://arxiv.org/abs/2404.13076) | Self-preference bias, which is why a judge from a different model family (Opus) checks the Qwen-graded results |
