Utility scripts for migration and maintenance.

- `index_runs.py`: scans canonical run folders and generates:
  - `runs/registry/runs_index.csv`
  - `runs/registry/runs_index.md`
- `sync_from_tinker.py`: one-shot legacy sync from `tinker-cookbook` into this repository.
- `use_api_profile.sh`: shell helper for switching `TINKER_API_KEY`/`TINKER_BASE_URL` to a named profile.

Usage:

```bash
python3 tools/index_runs.py
python3 tools/sync_from_tinker.py
source tools/use_api_profile.sh --list
source tools/use_api_profile.sh primary
```
