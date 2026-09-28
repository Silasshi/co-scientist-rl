# Reviewer Standards Synthesis — D5 Realignment Phase 1A

*Generated 2026-04-25 by 2 parallel web-research agents.*

This document captures what real ML/AI reviewers actually focus on when judging research papers, so the D5 oracle abstraction structure and audit rubric can be derived from authentic standards rather than invented top-down.

The synthesis has two parts:

- **Part A — General ML/AI reviewer guidelines**: NeurIPS/ICLR/ICML reviewer forms, NSF proposal criteria, reviewer-bias literature
- **Part B — TTT-Discover-adjacent OpenReview records**: 7 papers (MTTT, ReST-MCTS, ReST^EM, Voyager, SCoRe, Guided-ReST, DeepEvolve) — what reviewers focus on in the test-time-training / search-with-LLMs / self-improvement subfield

---

## PART A — General ML/AI Reviewer Guidelines

### Source 1: NeurIPS 2025 Reviewer Form

URL: https://neurips.cc/Conferences/2025/ReviewerGuidelines

**Four canonical dimensions** (each rated 1-4: poor/fair/good/excellent):

1. **Quality** — "Is the submission technically sound? Are claims well supported (by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?"
2. **Clarity** — "Is the submission clearly written? Is it well organized? Does it adequately inform the reader?"
3. **Significance** — "Are the results important? Are others (practitioners or researchers) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way?"
4. **Originality** — "Are the tasks or methods new? Is the work a novel combination of well-known techniques? Is it clear how this work differs from previous contributions? Is related work adequately cited?"

Plus: Summary, Questions (3-5), Limitations & broader impact, Overall Score (6-point: Strong Reject → Strong Accept), Confidence (5-point), Ethical concerns flag.

NeurIPS 2025 explicitly migrated from 10-point to 6-point scoring to reduce ambiguity.

### Source 2: ICLR 2025 Reviewer Guide

URL: https://iclr.cc/Conferences/2025/ReviewerGuide

**Four key questions**:
1. "What is the specific question and/or problem tackled by the paper?"
2. "Is the approach well motivated, including being well-placed in the literature?"
3. "Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous."
4. "What is the significance of the work? Does it contribute new knowledge and sufficient value to the community?"

Reviewers also assess: "is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (theoretically, algorithmically, etc.)?"

### Source 3: ICML 2025 Reviewer Instructions

URL: https://icml.cc/Conferences/2025/ReviewerInstructions

**Sections**:
- Summary — main findings, results, algorithmic/conceptual ideas
- Claims and Evidence — five sub-questions on whether claims supported, methods appropriate, proofs checked, experiments designed, supplementary reviewed
- Relation to Prior Works — three sub-items
- Other Aspects — open commentary on **originality, significance, clarity** (with explicit guidance to be open-minded about diverse contribution types)
- Questions for Authors
- Ethical Issues
- Overall Recommendation (5-point)

Four canonical evaluation dimensions: **soundness, presentation, significance, originality**.

### Source 4: NeurIPS Paper Checklist (de facto reproducibility checklist)

URL: https://neurips.cc/public/guides/PaperChecklist

**16 verbatim categories** (each Yes/No/NA + justification):
1. Claims (abstract/intro accurately reflect contributions)
2. Limitations (separate Limitations section, robustness to assumptions)
3. Theory, Assumptions and Proofs (full assumption set + complete proofs)
4. Experimental Result Reproducibility
5. Open Access to Data and Code
6. Experimental Setting/Details (data splits, hyperparameters, optimizer)
7. Experiment Statistical Significance (error bars, definition, significance tests)
8. Experiments Compute Resources (type, memory, runtime)
9. Code of Ethics
10. Broader Impacts
11. Safeguards
12. Licenses
13. Assets
14. Crowdsourcing and Human Subjects
15. IRB Approvals
16. Declaration of LLM Usage (added 2024-2025)

### Source 5: NSF Merit Review Criteria (PAPPG 24-1, Chapter III)

URL: https://www.nsf.gov/policies/pappg/24-1/ch-3-proposal-processing-review

Two top-level criteria: **Intellectual Merit** and **Broader Impacts**. Both evaluated against five elements:
1. Potential to advance knowledge (within field / across fields) and benefit society
2. Whether activities suggest creative, original, or **potentially transformative** concepts
3. Whether the **plan is well-reasoned, well-organized, and based on sound rationale**, with mechanisms to assess success
4. Qualifications of the team
5. Adequacy of resources

NSF framing question for reviewers: "what proposers want to accomplish, why they want to do it, how they plan to proceed, how they will measure success, and what benefits could result if the project succeeds."

### Source 6: Stanford CS230 Project Rubric

URL: https://cs230.stanford.edu/project/

Three primary dimensions:
- **Technical Quality** — "Does the technical material make sense? Are the things tried reasonable?"
- **Significance** — "Did the authors choose an interesting or 'real' problem, or only a small toy problem?"
- **Novelty** — "Is this project applying a common technique to a well-studied problem, or is the problem or method relatively unexplored?"

### Source 7: Reviewer-bias literature

- **Cortes & Lawrence 2021** (https://arxiv.org/abs/2109.09774): NeurIPS 2014 retrospective. ~50% of reviewer-quality-score variance is subjective. Among rejected papers, scores correlate with eventual citations; among accepted, no correlation. Conclusion: peer review is good at identifying poor papers, poor at identifying good ones.
- **NeurIPS 2021 Consistency Experiment**: 23% of duplicated papers received inconsistent accept/reject; 50.6% of papers accepted by Committee 1 were rejected by Committee 2. Disagreement sources: borderline noise (48%), genuine value disagreements, missed methodology issues.
- **Stelmakh-Shah 2023 (Peer Reviews of Peer Reviews)**: Length bias — lengthened reviews score higher even when added content non-informative. Outcome bias — authors positively biased toward reviews recommending acceptance.
- **Stelmakh-Shah 2023 (Citation Bias)**: Citing a reviewer's own work raises that reviewer's score by ~0.23 points on a 5-point scale.

### Synthesis A — universal dimensions across general guidelines

| Dimension | Frequency | What to look for |
|---|---|---|
| Soundness / Technical Quality / Claim Support | 6/6 | Methods appropriate; proofs complete; experiments well-designed; claims supported |
| Significance / Impact | 6/6 | Real problem, not toy; advances state of the art; new knowledge |
| Originality / Novelty | 6/6 | New tasks/methods/framings; novel combinations; differentiation from prior work |
| Clarity / Presentation | 6/6 | Clear writing; well-organized; understandable |
| Relation to Prior Work | 5/6 | Approach well-placed in literature; missing citations identified |
| Reproducibility (compute/code/data/hparams) | 4/6 | Code+data released; data splits + hparams specified; compute stated; CIs reported |
| Limitations / Honest self-assessment | 4/6 | Authors honest about weaknesses; assumptions stated; robustness discussed |
| Feasibility / Plan well-reasoned | 3/6 (NSF, GRFP, Stanford) | Specific aims; mechanisms to assess success; credible timeline; resources adequate |
| Broader / Societal Impact | 3/6 | Negative impacts discussed; benefits named; ethics |
| Transformative / Creative concept | 2/6 (NSF, ICML) | Potentially paradigm-shifting; creative combinations; not incremental |

**Top 4 make-or-break dimensions** (in every rubric): Soundness, Significance, Originality, Clarity. Plus, for proposal/plan-specific: Feasibility with measurable success criteria.

### Synthesis A — what reviewers do NOT focus on (anti-patterns to penalize)

- Length / page count (length-bias literature shows length is a *failure mode*, not a quality signal)
- Section-heading taxonomy (no rubric mandates particular section names)
- Jargon density (NSF GRFP: "It's more important that all members understand your work than impressing one expert")
- Citation density alone (NSF GRFP: "Demonstrate resourcefulness rather than citation density")
- Formatting compliance (catches via desk-reject; not a quality dimension)
- Confidence of tone (NeurIPS rewards authors being honest about weaknesses)
- Number of experiments alone (Goodhart axis; ICML emphasizes appropriateness over breadth)

---

## PART B — TTT-Discover-Adjacent OpenReview Records

**TTT-Discover specifically (arxiv 2601.16175)**: NOT FOUND on OpenReview. Paper is too new (Feb 2026); ICLR 2026 review window had already closed; NeurIPS 2026 reviews not yet public. No archival venue submission visible. Best evidence is press coverage (VentureBeat, WinBuzzer) which is not peer review.

The 7 most-relevant OpenReview-reviewed adjacent papers in the same subfield (test-time training / search-with-LLMs / self-improvement):

### Adjacent Paper 1: MTTT — Learning to (Learn at Test Time) — ICLR 2024

URL: https://openreview.net/forum?id=l7n59aufeT

Top reviewer concerns:
- **Insufficient task breadth**: All 4 reviewers flagged confined-to-ImageNet evaluation. Reviewer DvfS: *"This paper motivates from the TTT perspective, but no TTT experiments are performed."*
- **Missing prior-art positioning**: Specific prior work uncited (MT3 2022) cited as novelty-killer
- **Wall-clock vs FLOPs gap**: "Gap between FLOPs and wall-clock time suggests insufficient runtime analysis"
- **Hand-waving on optimization choices**: "Why is mini-batch SGD applicable for non-iid tokens?"

**Decisive criterion**: Failed because the gap between method narrative ("TTT") and what was actually tested. Reviewers refused the framing without tasks that genuinely required it.

### Adjacent Paper 2: ReST-MCTS* — NeurIPS 2024

URL: https://openreview.net/forum?id=8rcFOqEud5

Top reviewer concerns:
- **Theoretical correctness of value-update recursion**: Reviewer S8hx interrogated v_k bounds; reviewers refused to let the algorithm hand-wave guarantees
- **Algorithm-vs-search distinction**: "Algorithm resembles multi-step greedy rather than true heuristic search" — reviewers actively challenge whether claimed search behavior is real
- **Missing comparisons to direct competitors**: "Lack of comparison with AlphaLLM"
- **Stability under simultaneous training**: "Simultaneously training reward and policy can lead to instability"
- **Scalability**: Whether MCTS scaffolding survives at scale

Accepted only after concrete head-to-head comparisons + closing theoretical gaps.

### Adjacent Paper 3: ReST^EM (Beyond Human Data) — TMLR

URL: https://openreview.net/forum?id=lNAyUngGFK

Top reviewer concerns:
- **Saturation framing**: "Self-training improvements saturate after few iterations; potential overfitting" — reviewers want to know what limits the loop
- **Statistical rigor**: "No confidence intervals across experimental results" — single-run point estimates do not pass muster
- **Conceptual novelty**: "The idea of finetuning on model-generated data is a bit obvious" — for self-improvement papers, reviewers actively probe whether contribution is ≥ algorithmic insight
- **Hidden cost**: "The proposed method still requires numerous human prompts and a manual function to verify correctness"
- **Connection to existing theory**: Public commenter Lili Mou asked whether *"this paper represents a special case of REINFORCE."*

Accepted because the EM connection was made explicit. Without theoretical hook, "obvious" empirical loop was at risk.

### Adjacent Paper 4: Voyager — TMLR (NeurIPS FMDM oral)

URL: https://openreview.net/forum?id=ehfRiF0R3a

Top reviewer concerns:
- **"Whose learning is it?"**: Reviewer eudD: *"Voyager does not really learn, Chat-GPT does."* For test-time-adaptation papers, reviewers explicitly try to disentangle which component is doing the learning. **Most TTT-Discover-relevant critique on this list.**
- **Generalization beyond chosen domain**: All 3 reviewers asked about non-Minecraft domains
- **API leakage / abstraction**: "High-level action API abstracts away a large part of Minecraft" — reviewers check whether framework solves the hard part
- **Pretraining-prior dependence**: "Concerns about dependence on LLM knowledge encoded during pretraining"

Accepted only after agreeing the framework (curriculum + skill library + iterative prompting) is genuinely novel.

### Adjacent Paper 5: SCoRe (Self-Correction RL) — NeurIPS oral

URL: https://openreview.net/forum?id=CjwERcAU7w

Top reviewer concerns:
- **Reward specification gaming**: "Reward shaping with α>1 could incentivize the model to introduce minor errors in first steps to enable correction" — reviewers actively look for adversarial reward-hacking pathways
- **Simpler-baseline question**: "Missing comparison to simple baseline of end-to-end RL with correction prompt"
- **Reproducibility (closed model)**: "Experiments conducted on private Gemini models; hard to reproduce" (TTT-Discover avoids this — gpt-oss-120b)
- **Generalization dim count**: "No evaluation on diverse subjects like physics or chemistry"

Accepted as oral. Decisive factor: rigorous failure-mode analysis showed authors understood *why* prior methods failed before proposing fix.

### Adjacent Paper 6: Guided-ReST (Stream-of-Search + ReST)

URL: https://openreview.net/forum?id=Awl3ZhDHuQ

Top reviewer concerns:
- **Compute accounting must be real**: "No actual latency or compute cost accounting; claims based only on token budgets"
- **Single-task problem**: Countdown evaluation does not generalize
- **Optimal-trace dependence**: "Approach for generating training data relies on having ground truth solutions"
- **No error bars**: "Only average results reported; no error bands"

**Rejected** (3/5/4/5). Single task + missing compute accounting + missing CIs were terminal.

### Adjacent Paper 7: DeepEvolve

URL: https://openreview.net/forum?id=zUkBSNDrMx

Top reviewer concerns:
- **Per-domain evidence required**: "For any LLM-involved workflow, it is nearly guaranteed to receive better performance if we just replace the LLM included with deep research." **Pipeline-stacking failure mode.**
- **Statistical rigor**: "No standard deviations, p-values, or multiple-run averages"
- **Per-domain ablations**
- **Comparison to SOTA solvers, not just baselines**
- **LLM-as-judge unverified**

**Rejected**. Authors didn't engage with reviewer questions in rebuttal.

### Synthesis B — subfield-specific dimensions

| Dimension | Freq across 7 | Strong vs weak treatment |
|---|---|---|
| **Task breadth beyond chosen demo** | 7/7 | **Strong**: ≥3 distinct task families, ≥1 non-toy. **Weak**: Single benchmark |
| **Statistical rigor (CIs, error bars, n-seeds)** | 5/7 | **Strong**: Error bands + multi-seed + p-values. **Weak**: Single-run point estimates |
| **"Whose learning is it?" disentanglement** | 4/7 | **Strong**: Ablations swapping LLM, freezing components, comparing pretraining-only baselines. **Weak**: Treating LLM+method as one black box |
| **Compute / wall-clock accounting** | 4/7 | **Strong**: GPU-hours, $/run, ms/token. **Weak**: FLOP counts only |
| **Reduction to known primitives** | 4/7 | **Strong**: Method derived as special case of existing primitive. **Weak**: "Novel" but reduces to known algorithm |
| **Reward-hacking / adversarial pathway analysis** | 3/7 | **Strong**: Authors identify possible Goodhart routes and show they don't occur. **Weak**: No adversarial probe |
| **Comparison to simpler baseline ("just do X")** | 4/7 | **Strong**: Show simpler version + quantify gap. **Weak**: Compare only to historical baselines |
| **Theoretical justification of optimization choices** | 3/7 | **Strong**: Rationale for SGD vs Adam, mini-batch composition, learning rate. **Weak**: "Standard hyperparameters" |
| **Saturation / iteration limits** | 3/7 | **Strong**: Run loop until plateau, characterize, explain. **Weak**: Stop at convenient n with monotonic claims |
| **LLM-as-judge bias control** | 2/7 | **Strong**: Validate against human ratings or held-out judge. **Weak**: LLM judge no calibration |
| **Reproducibility (open model + code)** | 3/7 | **Strong**: Open weights, public code, deterministic seeds. **Weak**: Closed/private base model |

### Synthesis B — non-negotiable go/no-go criteria

In this subfield, reviewers treat as essentially required:

1. **At least one task where the method's framing is *necessary*** (not just sufficient). MTTT failed exactly this — claimed TTT but tested only ImageNet.
2. **Multi-seed statistical reporting**. Single-run claims are challenged.
3. **Ablation disentangling base-model capability from method**. Frozen-LLM and base-LLM comparisons required.
4. **Compute/cost accounting in operational units** ($, GPU-hours, ms/token).
5. **Head-to-head against the strongest direct competitor** (not just untrained baseline).
6. **Reward / search-objective robustness analysis**. SCoRe accepted *because* of failure-mode work.
7. **Saturation / scaling characterization**. What limits the loop?
8. **Generality story beyond the four headline domains**.

The single most likely critique TTT-Discover-adjacent papers face: **"is RL at test time *necessary*, or does best-of-N on a frozen model match it?"** — the natural simpler-baseline reviewers always demand.

---

## Combined synthesis (universal + subfield, the basis for D5 oracle + audit design)

### Two-layer framework matches user's hybrid rubric vision

**Universal layer (from Part A)** — applies to any research paper, regardless of subfield:
1. Soundness / Claim support
2. Significance / Impact
3. Originality / Novelty
4. Clarity / Presentation
5. Reproducibility (CIs, code, hparams, compute)
6. Limitations / Honest self-assessment
7. Feasibility / Well-reasoned plan with measurable success criteria (proposal-specific addition)

**Subfield-specific layer (from Part B, for test-time-discovery / search-with-LLMs / self-improvement papers)**:
1. Task breadth — ≥1 task where the framing is *necessary*
2. Disentanglement of LLM-prior vs method contribution
3. Compute/cost accounting in real units
4. Comparison to strongest direct competitor + simpler baseline
5. Reward-hacking analysis
6. Saturation / iteration-limit characterization
7. Reduction to known primitives + theoretical hook

### Anti-patterns (penalize regardless of layer)

- Long without substance
- Section-heading scaffolds without inline mechanism
- Jargon dense without explanation
- Citation count without appropriate placement
- Generic algorithm names without instantiation ("REINFORCE", "LoRA fine-tuning")
- Vague hyperparameters ("small learning rate", "appropriate batch size")
- Single-task demonstration claiming generality
- LLM-as-judge without calibration
- Closed-model / non-reproducible methodology

---

## Implications for Oracle Abstraction structure

The oracle abstraction (extracted from real bibliography papers) should map cleanly onto these reviewer-focus areas. Each bibliography paper extraction should yield content that the final-plan-generator can use to satisfy reviewer concerns:

- Methodology insights → satisfies Soundness + reduction-to-primitives
- Theoretical framing → satisfies Originality + theoretical hook
- Math / formalism → satisfies Soundness + reduction-to-primitives + reward-hacking
- Empirical methodology → satisfies Reproducibility + statistical rigor
- Failure modes / limitations from prior work → satisfies Significance (what gap is being filled) + reward-hacking
- Compute / scale numbers → satisfies operational compute accounting

Proposed oracle category set (to be discussed with user in Phase 1B):
1. **Insights** (problem framing + key conceptual moves from each paper)
2. **Methodology** (concrete mechanisms reusable in this plan)
3. **Theoretical framing** (algorithm reductions, theoretical guarantees)
4. **Math / formalism** (objective functions, gradient forms, update rules)
5. **Empirical setup** (benchmarks, baselines, statistical methodology)
6. **Failure modes / limitations** (what didn't work, with attribution)

## Implications for Audit Rubric structure

Hybrid universal + goal-specific (matches user's vision):

**Universal layer** (4-5 dimensions, always evaluate):
- Soundness (claims supported by inline math/empirical evidence?)
- Significance (real problem, not toy?)
- Originality (novel beyond pretraining or trivial recombination?)
- Clarity (well-organized; mechanisms specified?)
- Reproducibility (compute, hparams, statistical methodology stated?)

**TTT-Discover-specific layer** (3-4 items for this goal):
- Necessity of framing (is test-time-RL necessary or does best-of-N match?)
- Disentanglement (does plan separate LLM-prior contribution from method contribution?)
- Compute accounting (concrete $/GPU-hours stated?)
- Reward-hacking analysis (does plan address Goodhart pathways?)

Both proposed structures will be drafted as `ORACLE_DESIGN_v1.md` and `AUDIT_RUBRIC_v3.md`, then iterated with user.
