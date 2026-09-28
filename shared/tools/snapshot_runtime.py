#!/usr/bin/env python3
"""
Capture reproducibility snapshots for:
1) Conda environment "tinker"
2) Local tinker-cookbook git baseline

Outputs are written under <repo>/env/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in runtime output")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate + ("\n" if not candidate.endswith("\n") else "")


def snapshot_conda_env(repo_root: Path, conda_bin: str, env_name: str) -> None:
    env_dir = repo_root / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        env_dir / f"environment_{env_name}.yml",
        run([conda_bin, "env", "export", "-n", env_name, "--no-builds"]),
    )
    write_text(
        env_dir / f"conda_explicit_{env_name}.txt",
        run([conda_bin, "list", "-n", env_name, "--explicit"]),
    )
    write_text(
        env_dir / f"pip_freeze_{env_name}.txt",
        run([conda_bin, "run", "-n", env_name, "python", "-m", "pip", "freeze"]),
    )

    py_script = r'''
import json, platform, sys, importlib
pkgs = ['torch', 'numpy', 'datasets', 'tinker', 'chz', 'transformers', 'accelerate']
out = {
    'python_version': sys.version.replace('\\n', ' '),
    'python_executable': sys.executable,
    'platform': platform.platform(),
    'platform_node': platform.node(),
}
for name in pkgs:
    try:
        m = importlib.import_module(name)
        out[f'{name}_version'] = getattr(m, '__version__', 'unknown')
    except Exception as e:
        out[f'{name}_version'] = f'not-installed ({type(e).__name__})'
try:
    import torch
    out['torch_cuda_available'] = bool(torch.cuda.is_available())
    out['torch_cuda_version'] = getattr(torch.version, 'cuda', None)
    out['torch_num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    pass
print(json.dumps(out, indent=2))
'''
    runtime_raw = run([conda_bin, "run", "-n", env_name, "python", "-c", py_script])
    write_text(env_dir / f"runtime_snapshot_{env_name}.raw.txt", runtime_raw)
    runtime_json = extract_json_object(runtime_raw)
    write_text(env_dir / f"runtime_snapshot_{env_name}.json", runtime_json)


def snapshot_tinker_repo(repo_root: Path, tinker_repo: Path) -> None:
    env_dir = repo_root / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    write_text(env_dir / "tinker_repo_remote.txt", run(["git", "-C", str(tinker_repo), "remote", "-v"]))
    write_text(env_dir / "tinker_repo_status.txt", run(["git", "-C", str(tinker_repo), "status", "--porcelain=v1", "-b"]))

    def git(cmd: list[str]) -> str:
        return run(["git", "-C", str(tinker_repo)] + cmd).strip()

    snap = {
        "repo_path": str(tinker_repo),
        "head": git(["rev-parse", "HEAD"]),
        "head_short": git(["rev-parse", "--short", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "describe": git(["describe", "--tags", "--always", "--dirty"]),
        "origin_main_ref": git(["rev-parse", "origin/main"]),
        "merge_base_head_origin_main": git(["merge-base", "HEAD", "origin/main"]),
        "head_commit_detail_raw": git(["show", "-s", "--format=%H%n%h%n%an%n%ae%n%ad%n%s", "--date=iso-strict", "HEAD"]),
        "earliest_local_reflog_entry": run(["bash", "-lc", f"git -C '{tinker_repo}' reflog --date=iso | tail -1"]).strip(),
    }
    write_text(env_dir / "tinker_repo_snapshot.json", json.dumps(snap, indent=2) + "\n")


def write_env_doc(repo_root: Path, env_name: str) -> None:
    env_dir = repo_root / "env"
    runtime_path = env_dir / f"runtime_snapshot_{env_name}.json"
    runtime_raw_path = env_dir / f"runtime_snapshot_{env_name}.raw.txt"
    runtime = {}
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            runtime = {}
    if not runtime and runtime_raw_path.exists():
        try:
            runtime = json.loads(extract_json_object(runtime_raw_path.read_text(encoding="utf-8")))
        except Exception:
            runtime = {}

    lines = []
    lines.append("# Environment Lock\n\n")
    lines.append(f"Environment name: `{env_name}`\n\n")
    lines.append("## Runtime Summary\n")
    if runtime:
        for k in [
            "python_version",
            "python_executable",
            "platform",
            "torch_version",
            "torch_cuda_version",
            "torch_cuda_available",
            "torch_num_gpus",
            "numpy_version",
            "datasets_version",
            "tinker_version",
            "transformers_version",
        ]:
            if k in runtime:
                lines.append(f"- `{k}`: `{runtime[k]}`\n")
    else:
        lines.append("- Runtime snapshot unavailable.\n")

    lines.append("\n## Lock Files\n")
    lines.append(f"- `env/environment_{env_name}.yml`\n")
    lines.append(f"- `env/conda_explicit_{env_name}.txt`\n")
    lines.append(f"- `env/pip_freeze_{env_name}.txt`\n")
    lines.append(f"- `env/runtime_snapshot_{env_name}.json`\n")
    lines.append(f"- `env/runtime_snapshot_{env_name}.raw.txt`\n")
    lines.append("- `env/tinker_repo_snapshot.json`\n")
    lines.append("- `env/tinker_repo_status.txt`\n")
    lines.append("- `env/tinker_repo_remote.txt`\n")

    lines.append("\n## Refresh Command\n")
    lines.append("```bash\n")
    lines.append("python3 tools/snapshot_runtime.py\n")
    lines.append("```\n")

    write_text(repo_root / "docs" / "ENVIRONMENT.md", "".join(lines))


def write_repo_baseline_doc(repo_root: Path) -> None:
    snap_path = repo_root / "env" / "tinker_repo_snapshot.json"
    snap = {}
    if snap_path.exists():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Tinker Baseline\n\n")
    lines.append("This records the exact `tinker-cookbook` baseline used by this project.\n\n")

    for k in [
        "repo_path",
        "branch",
        "head",
        "head_short",
        "describe",
        "origin_main_ref",
        "merge_base_head_origin_main",
        "earliest_local_reflog_entry",
    ]:
        if k in snap:
            lines.append(f"- `{k}`: `{snap[k]}`\n")

    if "head_commit_detail_raw" in snap:
        lines.append("\n## HEAD Commit Detail\n\n")
        lines.append("```text\n")
        lines.append(snap["head_commit_detail_raw"] + "\n")
        lines.append("```\n")

    lines.append("\n## Notes\n")
    lines.append("- Use this baseline when comparing behavior after upstream updates.\n")
    lines.append("- Before adopting new upstream commits, re-run `tools/snapshot_runtime.py` and compare diffs.\n")

    write_text(repo_root / "docs" / "TINKER_BASELINE.md", "".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot conda env + tinker repo baseline")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--conda-bin", type=str, default="/home/silas/miniconda3/bin/conda")
    parser.add_argument("--env-name", type=str, default="tinker")
    parser.add_argument("--tinker-repo", type=Path, default=Path("/home/silas/tinker-cookbook"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    snapshot_conda_env(repo_root, args.conda_bin, args.env_name)
    snapshot_tinker_repo(repo_root, args.tinker_repo.resolve())
    write_env_doc(repo_root, args.env_name)
    write_repo_baseline_doc(repo_root)

    print("Snapshot complete")
    print(f"Repo: {repo_root}")
    print(f"Env docs: {repo_root / 'docs' / 'ENVIRONMENT.md'}")
    print(f"Baseline docs: {repo_root / 'docs' / 'TINKER_BASELINE.md'}")


if __name__ == "__main__":
    main()
