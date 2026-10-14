"""Standard-library-only checks against reintroducing private data or notebook output.

Run ``python tools/repository_data_policy.py`` from any directory in this clone.
The check is also part of the backend pytest suite; no pre-commit hook is needed.
Only paths and violation labels are printed, never notebook source or output.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_METADATA = {"execution", "ExecuteTime", "executionInfo", "widgets"}
PRIVATE_ROOTS = ("platform/data/real/", ".dvc/", ".review/", "docs/research-notes/", "docs/reports/")
PRIVATE_FILES = {"xops-service-checklist.md", "docs/xops-service-checklist.md",
                 "docs/plans/2026-09-16-데이터분석보고서-제작계획.md"}


def private_path(path: str) -> bool:
    return (path.startswith(PRIVATE_ROOTS) or path in PRIVATE_FILES or path.endswith(".dvc")
            or ("/" in path and path.split("/", 1)[0].startswith("data_version")))


def clean_notebook(document: dict) -> dict:
    """Clear execution products only; retain source, order and descriptive metadata."""
    cleaned = deepcopy(document)
    for cell in cleaned.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        cell.pop("attachments", None)
        for key in EXECUTION_METADATA:
            cell.get("metadata", {}).pop(key, None)
    for key in EXECUTION_METADATA:
        cleaned.get("metadata", {}).pop(key, None)
    return cleaned


def notebook_violations(document: dict) -> list[str]:
    violations = []
    if (not isinstance(document, dict) or document.get("nbformat") != 4
            or not isinstance(document.get("nbformat_minor"), int)
            or document["nbformat_minor"] < 0
            or not isinstance(document.get("cells"), list)
            or not isinstance(document.get("metadata"), dict)):
        return ["invalid notebook structure"]
    if EXECUTION_METADATA & document.get("metadata", {}).keys():
        violations.append("notebook execution metadata")
    for index, cell in enumerate(document.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
            violations.append(f"cell {index}: invalid cell structure")
            continue
        source = cell.get("source")
        if not (isinstance(source, str) or isinstance(source, list) and all(isinstance(s, str) for s in source)):
            violations.append(f"cell {index}: invalid source structure")
        if not isinstance(cell.get("metadata"), dict):
            violations.append(f"cell {index}: invalid metadata structure")
            continue
        if cell["cell_type"] == "code" and not isinstance(cell.get("outputs"), list):
            violations.append(f"cell {index}: invalid outputs structure")
        if cell.get("outputs"):
            violations.append(f"cell {index}: saved outputs")
        if cell.get("execution_count") is not None:
            violations.append(f"cell {index}: execution_count")
        if cell.get("attachments"):
            violations.append(f"cell {index}: attachments")
        if EXECUTION_METADATA & cell.get("metadata", {}).keys():
            violations.append(f"cell {index}: execution metadata")
    return violations


def tracked_paths(root: Path = ROOT) -> list[str]:
    data = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    return [path for path in data.decode("utf-8").split("\0") if path]


def repository_violations(root: Path = ROOT) -> list[str]:
    violations = []
    for path in tracked_paths(root):
        if private_path(path):
            violations.append(f"{path}: private path is tracked")
        if path.endswith(".ipynb"):
            for location in ("working tree", "index"):
                try:
                    data = ((root / path).read_bytes() if location == "working tree" else
                            subprocess.check_output(["git", "-C", str(root), "show", f":{path}"]))
                    document = json.loads(data)
                    violations.extend(f"{path} [{location}]: {issue}" for issue in notebook_violations(document))
                except (OSError, ValueError, TypeError, AttributeError, subprocess.CalledProcessError):
                    violations.append(f"{path} [{location}]: cannot validate notebook")
    return violations


def main() -> int:
    violations = repository_violations()
    if violations:
        print("\n".join(violations))
        return 1
    count = sum(path.endswith(".ipynb") for path in tracked_paths())
    print(f"PASS: private paths untracked; {count} notebooks have no saved execution products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
