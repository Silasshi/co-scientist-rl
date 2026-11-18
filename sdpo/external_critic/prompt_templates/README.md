<!-- D6 prompts directory — single source of truth for D6 production prompts. -->

# D6 Prompts

All D6 production prompts live here as `.md` files. Source code in
`src/co_scientist/d6_grant_proposal/` loads them via the loader at
`src/co_scientist/d6_grant_proposal/prompt_loader.py`.

**Editing a `.md` here changes the next training run.** No code change needed
unless you add a new template variable.

## Layout

| Folder       | Role                                                                 |
| ------------ | -------------------------------------------------------------------- |
| `generation/`| Policy prompts (student / teacher) and the cold-start critique placeholder |
| `review/`    | Reviewer (critic) instructions for the privileged-observation Opus subagent |
| `audit/`     | Absolute-quality scoring prompt for the multi-plan batch audit       |
| `pairwise/`  | Pairwise (A vs B) comparison prompt for close-cluster corroboration  |
| `oracle/`    | Oracle-item extraction prompt used by `extract_oracle.py`           |

## File format

Each template file looks like:

```markdown
<!-- optional metadata (source ref, role, vars) -->

# Title

**Role**: ...
**Template variables**: `{var1}`, `{var2}`

---

<actual template body — what `load_prompt` returns>
```

The loader returns everything after the first `---` rule line. Trailing
newlines are stripped so `.format(**kwargs)` produces bytes identical to the
previous hardcoded Python strings.

## Loader API

```python
from co_scientist.d6_grant_proposal.prompt_loader import (
    load_prompt,        # raw template string (no .format applied)
    render_prompt,      # load_prompt + str.format(**kwargs)
    reload_prompts,     # invalidate the lru_cache after editing a file mid-session
    PROMPTS_ROOT,
)
```

## Out-of-scope (intentional)

`build_single_call_prompt` (used by `train_grpo.py` and `train_baseline.py`)
is defined in `src/co_scientist/shared/grant_signal_reward.py` and is also used
by D4. To keep D5/D4 untouched, that prompt was NOT externalized in the
2026-04-30 D6 prompt centralization. If D6 ever forks its own single-call
audit prompt, add it under `audit/single_call_score.md`.
