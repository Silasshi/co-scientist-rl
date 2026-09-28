# projects/

Each research direction has its own project folder with the following structure:

```
projects/{direction}/
  README.md          # Direction overview, key results, entry points
  configs/           # Run configuration files
  runs/              # Training run artifacts
  analysis/          # Direction-specific analysis and results
  knowledge/         # Findings, method docs, design docs
  data/              # Direction-specific data files
  tools/             # Direction-specific utility scripts (if any)
```

## Current directions

- `rubric_reward/` — D1: GRPO with rubric reward (bestversion 0.693)
- `ibt/` — D2: Iterative Brainstorming Training
- `ttt_discover/` — D3: TTT-Discover with multi-signal reward

See `../DIRECTIONS.md` for the full map.
