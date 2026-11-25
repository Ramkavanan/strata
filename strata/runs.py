"""Numbered, timestamped run directories with config snapshots with config snapshots, so a result can be traced to the configuration that produced it."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Optional

from . import __version__
from .schema import RunManifest


def git_sha(cwd: Optional[str] = None) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _num(name: str) -> int:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else -1


def list_runs(root: str, kind: str) -> list[str]:
    d = os.path.join(root, kind)
    if not os.path.isdir(d):
        return []
    return sorted((os.path.join(d, n) for n in os.listdir(d) if _num(n) >= 0), key=lambda p: _num(os.path.basename(p)))


def resolve_run(root: str, kind: str, number: Optional[int] = None) -> str:
    runs = list_runs(root, kind)
    if not runs:
        raise FileNotFoundError(f"no {kind} runs under {root}")
    if number is None:
        return runs[-1]
    for r in runs:
        if _num(os.path.basename(r)) == number:
            return r
    raise FileNotFoundError(f"{kind} run {number} not found")


def new_run(
    root: str,
    kind: str,
    config: dict,
    parent: Optional[dict] = None,
    snapshot: Optional[list[str]] = None,
    hashes: Optional[dict] = None,
) -> tuple[str, RunManifest]:
    runs = list_runs(root, kind)
    n = (_num(os.path.basename(runs[-1])) + 1) if runs else 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = os.path.join(root, kind, f"{n}_{ts}")
    os.makedirs(path)
    man = RunManifest(
        run_type=kind,
        run_number=n,
        timestamp=ts,
        harness_version=__version__,
        git_sha=git_sha(),
        parent=parent or {},
        config=config,
        hashes=hashes or {},
    )
    for src in snapshot or []:
        dst = os.path.join(path, "config_snapshot", os.path.basename(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.exists(src):
            shutil.copy2(src, dst)
    write_manifest(path, man)
    return path, man


def write_manifest(path: str, man: RunManifest) -> None:
    with open(os.path.join(path, "manifest.json"), "w") as f:
        f.write(man.model_dump_json(indent=2))


def read_manifest(path: str) -> RunManifest:
    with open(os.path.join(path, "manifest.json")) as f:
        return RunManifest.model_validate_json(f.read())


def dump_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
