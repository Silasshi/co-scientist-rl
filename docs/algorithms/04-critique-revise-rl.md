# Test-time RL with an entropic objective, plus critique–revise (D3 / D4)

**Follows:** TTT-Discover, *Learning to Discover at Test Time*, [arXiv:2601.16175](https://arxiv.org/abs/2601.16175). It does per-problem RL with an entropic (max-reward-seeking) objective over a buffer of attempts.

## Entropic objective — [`ttt_discover/train_buffer_ttt.py`](../../src/co_scientist/ttt_discover/train_buffer_ttt.py)
The aim is to find *one* excellent plan for a single goal, not to raise the average. So samples are weighted by `softmax(β·R)` instead of `R − mean`.
- β is found by binary search so that `KL(softmax(βR) ‖ uniform)` hits a fixed budget ([`adaptive_beta` L527](../../src/co_scientist/ttt_discover/train_buffer_ttt.py#L527)). This makes the concentration on top samples adaptive instead of hand-tuned.
- The weights are rescaled to mean 1 so they can be used directly as PPO advantages ([`entropic_weights` L552](../../src/co_scientist/ttt_discover/train_buffer_ttt.py#L552)).
- A buffer of past plans gives context to later generations ([`select_context` L245](../../src/co_scientist/ttt_discover/train_buffer_ttt.py#L245)).

## Critique–revise — [`ttt_discover/train_critique_revise.py`](../../src/co_scientist/ttt_discover/train_critique_revise.py)
When all the top plans tied on the 1–5 integer signal scale, the entropic objective plateaued. So each iteration adds a targeted revision phase:

1. **Explore:** sample fresh plans conditioned on the buffer.
2. **Exploit:** pick top plans by UCB ([`ucb_select` L296](../../src/co_scientist/ttt_discover/train_critique_revise.py#L296)), find each plan's weakest signal ([`identify_bottleneck` L323](../../src/co_scientist/ttt_discover/train_critique_revise.py#L323)), and generate a critique and revision aimed at that signal.
3. **Update:** fresh plans get entropic weights. Revisions get a delta-weighted advantage: 0.7 × the change on the targeted signal + 0.3 × the change in the total ([config L135–L143](../../src/co_scientist/ttt_discover/train_critique_revise.py#L135)).

The grant-proposal port, [`grant_proposal/train_cr_v7.py`](../../src/co_scientist/grant_proposal/train_cr_v7.py), adds:
- per-signal REINFORCE
- n-gram novelty and a GAPO-style diversity bonus ([L87–L128](../../src/co_scientist/grant_proposal/train_cr_v7.py#L87))
- tracking of grader confidence
- OpenAlex retrieval

## The key ablation: how much does RL add?
Every component was ablated on the same goal, with an independent Opus judge alongside the training grader ([`projects/ttt_discover/STATUS.md`](../../projects/ttt_discover/STATUS.md)):

| Condition | Training grader (Qwen) | Opus / 20 |
|---|---|---|
| Critique–revise, no RL, best-of-2 | 0.950 | 9 |
| Critique–revise + RL, best-of-2 | 0.950 | 9 |
| Critique–revise + RL on fresh plans, best-of-1 | 0.927 | 10 |
| Critique–revise, no RL, best-of-1 | 0.840 | 8 |
| Revision **without** the per-signal critique | 0.763 | 6 |

Reading:
- Most of the gain comes from **conditioning on the critique**: +0.19 with RL held fixed.
- RL helps only when it trains the *exploration* policy (+0.087 at best-of-1). Even then, RL at best-of-1 roughly equals no-RL at best-of-2.

On grant proposals (D4) the same RL loop pushed the proxy reward to 0.970, while Opus rated the best plan 11 / 20. The no-RL critique–revise baseline got 16 / 20 ([`projects/grant_proposal/STATUS.md`](../../projects/grant_proposal/STATUS.md)). The diagnosis was **template collapse**, not a wrong reward: across conditions the proxy still ranked plans like Opus (ρ = +0.91, [`DECISIONS.md` L49](../../projects/grant_proposal/DECISIONS.md#L49)), but RL concentrated the policy on one template.
