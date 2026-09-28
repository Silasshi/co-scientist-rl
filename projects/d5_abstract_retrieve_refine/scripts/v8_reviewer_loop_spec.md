# v8 Batch 1+2+3 Reviewer Loop — Master Spec

**Filed**: 2026-04-28 by Claude Code session before user sleep
**Auto-fires**: 2026-04-28 06:40 BST via single-shot CronCreate
**Total compute**: ~12 cells × ~6 hr = ~72 hr wall (sequential, server constraint)
**Total cost**: ~$615 (Tinker + critic + audit Opus subagents)

---

## You are the Reviewer

This Claude Code session has been fired automatically because user's quota reset 2 hours after they went to sleep. You are responsible for:

1. **Launching the orchestrator** — a bash script that runs 12 cells sequentially
2. **Acting as in-process critic + audit subagent dispatcher** while training runs
3. **Auto-running final M8-strict audit** on all eval_rollouts after all 12 cells complete
4. **Comparing against σ_v8 anchor only** — NOT σ legacy 25.25 (footer/methodology mismatch)

User went to sleep saying: "我希望你直接启动subagent,按照同样的流程帮我做reviewer,因为我觉得在另一个session开daemon,它的一些反馈你看不到,所以就没办法做及时的调整"

So ALL critic + audit subagent dispatches happen INSIDE this session via the Agent tool, not via separate daemon Claude window.

---

## Step 1 — Pre-flight (do this first thing)

Read these files:
- `projects/d5_abstract_retrieve_refine/STATUS.md` (last entry = current pre-cron context)
- `projects/d5_abstract_retrieve_refine/knowledge/current/RUN_REGISTRY.md` (anchor numbers)
- `projects/d5_abstract_retrieve_refine/knowledge/current/AUDIT_RUBRIC_v3.md` (9-dim rubric for audit subagents)
- `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/AUDIT_SUMMARY.md` (σ_v8 = 19.38/45 anchor)
- `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_base_truncated_aborted/ABORTED_README.md` (why prior base was killed)
- `projects/d5_abstract_retrieve_refine/scripts/run_v8_batch1_orchestrator.sh` (the script you'll launch)

Verify:
- `train_mu_v8_d5sdpo.py` Config default `max_tokens: int = 8192` (not 4096)
- σ_v8 buffer/audit exists at `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/`
- No active trainer process: `pgrep -af "train_mu_v8_d5sdpo"` should be empty
- `_v8_batch1_ALL_DONE.flag` does NOT yet exist in `runs/`

---

## Step 2 — Launch orchestrator + reviewer loop

Launch the bash orchestrator in the background. It will sequentially launch 12 trainers, each blocking on `critic.wait()` per iter.

```bash
nohup bash projects/d5_abstract_retrieve_refine/scripts/run_v8_batch1_orchestrator.sh \
    > projects/d5_abstract_retrieve_refine/runs/_v8_orchestrator.log 2>&1 &
disown
```

Now enter the **reviewer loop**: the orchestrator's trainer is blocked waiting for critic responses. You must dispatch them.

---

## Step 3 — Reviewer poll cycle (use /loop dynamic mode, ~120s heartbeat)

Each /loop cycle:

### 3a. Find pending critic requests
```bash
PENDING_CRITIC=$(find projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_*/critic_requests/ \
    -name "iter_*.json" 2>/dev/null | while read req; do
        resp=$(echo "$req" | sed 's|critic_requests|critic_responses|')
        [ ! -f "$resp" ] && [ ! -f "$req.done" ] && echo "$req"
    done)
```

For EACH pending critic request, dispatch ONE Agent subagent IN PARALLEL (multiple Agent calls in same assistant message). The critic prompt is at `specs/subagents/opus_critic_audit_daemon.md` lines 99-131 (CRITIC per-plan task spec). The subagent must:
1. Read the request JSON (`critic_requests/iter_NNN.json`)
2. Apply the critic system prompt with goal + source_paper_md + plan substituted
3. Output ONLY a `<critique>...</critique>` block
4. Write `critic_responses/iter_NNN.json` atomically with this schema:
   ```json
   {
     "iter": <int>,
     "kind": "critic",
     "completed_at": "<ISO-8601>",
     "judgments": {"<plan_id>": {"critique_xml": "<critique>...</critique>"}}
   }
   ```
5. Rename request `iter_NNN.json` → `iter_NNN.json.done`

Use `Agent(subagent_type="general-purpose", run_in_background=True, ...)` — don't block on each one. Multiple critics across different cells can run in parallel.

### 3b. Find pending audit requests (async, lower priority)
```bash
PENDING_AUDIT=$(find projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_*/audit_requests/ \
    -name "iter_*.json" 2>/dev/null | while read req; do
        resp=$(echo "$req" | sed 's|audit_requests|audit_responses|')
        [ ! -f "$resp" ] && [ ! -f "$req.done" ] && echo "$req"
    done)
```

For EACH pending audit request, dispatch 8 PARALLEL Agent subagents (one per plan, M8-compliant strict 1-plan/Opus). Use the **9-dim audit_v3 rubric** at `knowledge/current/AUDIT_RUBRIC_v3.md` (NOT the old 4-dim daemon spec). Each subagent:
1. Reads the audit request to get the plan it's assigned
2. Applies audit_v3 9-dim scoring (5 universal + 4 TTT-specific, /45 total)
3. Returns scores

After all 8 subagents return, aggregate into one response file:
```json
{
  "iter": <int>,
  "kind": "audit",
  "completed_at": "<ISO-8601>",
  "judgments": {
    "<plan_id>": {
      "universal_scores": {"U1_soundness": {"score":...}, ...},
      "subfield_scores": {"T1_necessity": {"score":...}, ...},
      "universal_total": <int>,
      "subfield_total": <int>,
      "total": <int>,  // /45 — trainer reads this for early-stop
      "weighted_total_norm20": <float>
    },
    ...
  }
}
```

Note: trainer's early_stop reads `judgments[*]["total"]`. Make sure each plan_id has a numeric `total` so the early-stop logic works at threshold 10.0.

### 3c. Check orchestrator status
```bash
# How many cells done?
test -f projects/d5_abstract_retrieve_refine/runs/_v8_batch1_ALL_DONE.flag && echo "ALL DONE"
# State file:
cat projects/d5_abstract_retrieve_refine/runs/_v8_orchestrator_state.json
```

If `_v8_batch1_ALL_DONE.flag` exists → break out of /loop, go to Step 4.
Else → ScheduleWakeup another 120s and continue.

---

## Step 4 — Final analysis (after all 12 cells done)

After orchestrator finishes:

### 4a. Pairwise tournament — use existing `mu_v4_pairwise_v1.py` infra forked per matchup

| Matchup | n pairs | Threshold | Purpose |
|---|---:|---|---|
| **G vs v7-opd-full** | 8 | ≥6/8 | PAPER HEADLINE |
| **base vs v7-opd-full** | 8 | ≥6/8 | gates Decision 1 (Batch 3 drop check — already moot since Batch 3 already ran) |
| **G vs base** | 8 | ≥6/8 | imports-vs-critique-fix marginal |
| **v7-opd-full vs σ_v8** | 8 | directional | Decision 2 control (separates length confound from critique-fix lift) |
| each cell vs σ_v8 | 8 each | directional | per-cell lift |
| G vs G+ | 8 | directional | α=0.05 vs α=0.1 |

### 4b. Strict 1-plan/Opus audit_v3 isolated — peak iter per cell

For each of 12 cells, identify peak iter (highest audit /45 in metrics.jsonl). Re-audit those 8 plans with strict 1-plan/Opus (M8-compliant) for paper-grade peak number. Use the same 8-Agent-parallel pattern as σ_v8 audit (see `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/audit_responses/`).

### 4c. Factorial decomposition

Per RUN_CONFIRMATION decision matrix:
```
mask main effect = mean(A,C,Cplus,E,G,Gplus pairwise vs σ_v8) - mean(base,B,Bplus,D,F,Fplus)
α main effect (3-level)  = mean over α=0 cells, α=0.05 cells, α=0.1 cells; check monotonicity
PPO main effect = mean(D,E,F,Fplus,G,Gplus) - mean(base,A,B,Bplus,C,Cplus)
α × cliff-iter regression = correlation of α with per-cell cliff iter
```

### 4d. Update RUN_REGISTRY.md

Add 13 new rows (sigma_v8 already there; base + A-G + Bplus/Cplus/Fplus/Gplus). Anchor numbers section: lock production checkpoint = cell with highest pairwise win rate vs σ_v8.

### 4e. Draft finding doc F15

`paper_materials/findings/F15_v8_d5sdpo_factorial_ablation.md` — capture:
- Headline: which cell wins, by what margin vs σ_v8 + v7-opd-full + μ-v4 nominal
- Critique-fix isolated effect (base vs v7-opd-full)
- 3 axis main effects + interactions
- Pre-registered hypothesis check (cliff-iter monotonic in α?)
- M8 corroboration (pairwise + strict /45 agreement)
- Production checkpoint pointer

### 4f. Send PushNotification to user

```
v8 Batch 1+2+3 complete: <N>/12 cells succeeded; production winner = <cell> (+<Δ> over σ_v8). F15 finding doc drafted.
```

---

## Critical reminders

1. **σ_v8 (19.38/45) is the ONLY length-matched anchor for v8 cells**. Do NOT compare to σ legacy 25.25 (different footer + audit methodology).
2. **v7-opd-full (26.62 strict /45 at iter 4)** uses 750-word footer; comparison to v8 cells is mixed-methodology. Use pairwise (M8-compliant, methodology-agnostic) as primary headline.
3. **μ-v4 nominal (28.00/45)** is M8-confounded; cite with caveat per RUN_REGISTRY:160.
4. If any cell trainer fails (rc != 0): orchestrator continues to next cell. Log failure, do NOT halt the whole run.
5. If critic timeout fires (>900s on a single iter): orchestrator's trainer falls back to COLD_START_CRITIQUE_NONE for that iter and continues. Track `critique_was_cold_start: true` rate per cell.
6. If you (the reviewer) hit subagent dispatch issues: the trainer's existing fallback to cold-start preserves correctness. Don't panic.

---

## State machine summary

```
[cron fires at 06:40] → [pre-flight checks] → [launch orchestrator (bash bg)]
                               ↓
                         [enter /loop with 120s heartbeat]
                               ↓
[poll critic_requests + audit_requests across 12 run dirs]
                               ↓
[dispatch Agent subagents in parallel for any pending requests]
                               ↓
[write critic_responses/audit_responses atomically; rename request to .done]
                               ↓
[check _v8_batch1_ALL_DONE.flag] → if false: ScheduleWakeup 120s, loop again
                               ↓
                          if true: → [Step 4: final pairwise + audit + F15 + PushNotification]
                               ↓
                          [orchestrator complete; loop ends]
```

This spec is self-contained. Execute it.
