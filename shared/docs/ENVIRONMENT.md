# Environment Lock

Environment name: `tinker`

## Runtime Summary
- `python_version`: `3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]`
- `python_executable`: `/home/silas/miniconda3/envs/tinker/bin/python`
- `platform`: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- `torch_version`: `2.9.1+cu128`
- `torch_cuda_version`: `12.8`
- `torch_cuda_available`: `False`
- `torch_num_gpus`: `0`
- `numpy_version`: `2.4.0`
- `datasets_version`: `4.4.2`
- `tinker_version`: `0.7.0`
- `transformers_version`: `4.56.2`

## Lock Files
- `shared/env/environment_tinker.yml`
- `shared/env/conda_explicit_tinker.txt`
- `shared/env/pip_freeze_tinker.txt`
- `shared/env/runtime_snapshot_tinker.json`
- `shared/env/runtime_snapshot_tinker.raw.txt`
- `shared/env/tinker_repo_snapshot.json`
- `shared/env/tinker_repo_status.txt`
- `shared/env/tinker_repo_remote.txt`

## Refresh Command
```bash
python3 tools/snapshot_runtime.py
```

## Named API Profiles

The trainers support named API profiles without replacing your existing `TINKER_API_KEY`.

Environment variable convention:
- `CO_SCIENTIST_API_KEY_<PROFILE>`: required secret for a named profile.
- `CO_SCIENTIST_BASE_URL_<PROFILE>`: optional endpoint override for that profile.
- `CO_SCIENTIST_ACTIVE_API_PROFILE`: optional default profile for the current shell/process.

Examples:
```bash
export CO_SCIENTIST_API_KEY_PRIMARY='...'
export CO_SCIENTIST_API_KEY_ALT='...'
export CO_SCIENTIST_BASE_URL_ALT='https://example.invalid/services'
```

Usage:
- Shell-wide switch: `source tools/use_api_profile.sh primary`
- List configured profiles: `source tools/use_api_profile.sh --list`
- Per-run switch: `python src/co_scientist/rubric_reward/sdpo/train_sdpo.py api_profile=alt`

If `api_profile` is unset, the code keeps using the ambient `TINKER_API_KEY`.
