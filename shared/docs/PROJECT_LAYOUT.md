# Project Layout

*Last updated: 2026-04-10*

## Directory Structure

```
co-scientist-project/
├── src/co_scientist/               # All source code
│   ├── shared/                     # Shared utilities (eval_core, openrouter_client, api_profiles)
│   ├── rubric_reward/              # Rubric-reward methods
│   │   ├── grpo/                   # Single-stage GRPO (bestversion, CPR, RC-GRPO, etc.)
│   │   ├── sdpo/                   # SDPO trainer
│   │   ├── refinement/             # Two-stage refinement (doublegeneration, rubric dropout)
│   │   ├── multiturn/              # Multi-turn V1-V4 trainers
│   │   ├── selector/               # Pairwise plan selector
│   │   └── reward_shaping/         # Reward-shaping ablations (weighted, hard_min, etc.)
│   ├── ibt/                        # Iterative brainstorming training
│   ├── ttt_discover/               # TTT-Discover / 7-signal methods
│   └── eval/                       # All evaluation scripts
│
├── projects/                       # Per-direction project artifacts
│   ├── rubric_reward/              # Rubric-reward direction
│   │   ├── configs/                # Config system (templates/ + run_configs/)
│   │   ├── data/                   # Training data (CPR pairs, curriculum)
│   │   ├── runs/                   # Run artifacts by year_month/method
│   │   ├── analysis/               # Direction-specific analysis
│   │   └── knowledge/              # Direction-specific knowledge
│   ├── ibt/                        # IBT direction
│   │   ├── configs/
│   │   ├── data/
│   │   ├── runs/
│   │   ├── analysis/
│   │   └── knowledge/
│   └── ttt_discover/               # TTT-Discover direction
│       ├── configs/
│       ├── data/
│       ├── runs/
│       ├── analysis/
│       ├── knowledge/
│       └── tools/
│
├── shared/                         # Cross-direction shared resources
│   ├── docs/                       # Project documentation
│   │   ├── AGENT.md                # Agent operational rules
│   │   ├── ASSUMPTIONS.md          # Core assumptions
│   │   ├── ENVIRONMENT.md          # Runtime environment
│   │   ├── EXPERIMENTS.md          # Method catalog and procedures
│   │   ├── PROJECT_LAYOUT.md       # This file
│   │   ├── STATUS.md               # Current research stage
│   │   ├── TINKER_BASELINE.md      # Framework baseline
│   │   ├── VERIFICATION_CHECKLIST.md # Reproducibility checklist
│   │   └── WHERE_TO_WORK.md        # Active vs read-only paths
│   ├── tools/                      # Utility scripts (index_runs, monitoring, etc.)
│   ├── knowledge/                  # Cross-direction experiment knowledge
│   │   ├── EXPERIMENT_CATALOG.md   # Master catalog of all experiments
│   │   └── RESEARCH_LOG.md         # Research log
│   ├── analysis/                   # Cross-direction analysis notebooks
│   │   ├── active/                 # Current research notebooks
│   │   ├── diagnosis/              # Systematic analysis
│   │   ├── comparison/             # Cross-method comparisons
│   │   ├── archive/                # Historical notebooks
│   │   ├── data/                   # Data exploration
│   │   └── exercise_silas/         # Learning exercises
│   ├── papers/                     # Reference papers
│   └── env/                        # Environment configuration
│
├── paper/                          # Paper and presentation materials
│   ├── overleaf/                   # LaTeX source (NeurIPS 2026)
│   ├── design/                     # Pipeline design documents
│   ├── outlines/                   # Paper narrative outlines
│   ├── scripts/                    # Figure/presentation generation
│   ├── presentation/               # Presentation examples
│   ├── reference/                  # NeurIPS formatting guidelines
│   └── archive/                    # Historical figures
│
├── tests/                          # Unit tests
├── ARCHIVE/                        # Pre-restructure code archive
│   └── legacy_snapshot/
└── DIRECTIONS.md                   # Overview of research directions
```

## Key Navigation Paths

| Goal | Path |
|------|------|
| Edit rubric-reward method code | `src/co_scientist/rubric_reward/<family>/` |
| Edit IBT method code | `src/co_scientist/ibt/` |
| Edit TTT-Discover code | `src/co_scientist/ttt_discover/` |
| Edit shared utilities | `src/co_scientist/shared/` |
| Edit eval scripts | `src/co_scientist/eval/` |
| View past runs (rubric-reward) | `projects/rubric_reward/runs/<year_month>/<method>/` |
| Understand findings | `shared/knowledge/EXPERIMENT_CATALOG.md` |
| Analysis notebooks | `shared/analysis/active/` |
| Paper materials | `paper/overleaf/` |
| Config templates | `projects/<direction>/configs/templates/` |

## Read-First Order for New Researchers

1. `README.md` -- Project overview
2. `DIRECTIONS.md` -- Research directions overview
3. `shared/docs/STATUS.md` -- Current stage and key numbers
4. `shared/knowledge/EXPERIMENT_CATALOG.md` -- All experiments and findings
5. `shared/docs/WHERE_TO_WORK.md` -- What to edit vs leave alone
6. `shared/docs/EXPERIMENTS.md` -- Method catalog and procedures
