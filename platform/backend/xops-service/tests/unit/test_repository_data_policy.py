"""Check retained notebooks and ignore rules without reading any external data."""
from pathlib import Path
import json
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[5]
POLICY = runpy.run_path(str(ROOT / "tools" / "repository_data_policy.py"))


def test_repository_does_not_reintroduce_private_paths_or_saved_notebook_outputs():
    assert POLICY["repository_violations"](ROOT) == []


def test_ignore_rules_cover_new_files_in_every_private_data_category():
    paths = ["platform/data/real/new.csv", "platform/data/real/nested/new.xlsx",
             ".dvc/config", ".review/verification.yaml", "xops-service-checklist.md",
             "docs/xops-service-checklist.md", "docs/research-notes/new.md",
             "data_version/new.csv", "data_version_2030/new.csv", "nested/source.dvc",
             "docs/reports/new.md", "docs/plans/2026-09-16-데이터분석보고서-제작계획.md"]
    result = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin", "-z"],
                            input=("\0".join(paths) + "\0").encode("utf-8"), capture_output=True, check=True)
    # NUL-delimited bytes avoid Windows text-mode CRLF conversion and Git path quoting.
    ignored = set(result.stdout.decode("utf-8").split("\0")) - {""}
    assert ignored == set(paths)
    assert all(POLICY["private_path"](path) for path in paths)


def test_notebook_cleanup_keeps_source_order_types_and_kernel_metadata():
    document = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3"},
                                           "language_info": {"name": "python"}, "widgets": {"state": "toy"}},
                "cells": [{"cell_type": "markdown", "source": ["example"], "attachments": {"image": "toy"},
                           "metadata": {"tags": ["keep"]}},
                          {"cell_type": "code", "source": ["print(1)"], "outputs": [{"text": ["1"]}],
                           "execution_count": 1, "metadata": {"execution": {"time": "toy"}, "tags": ["keep"]}}]}
    cleaned = POLICY["clean_notebook"](document)
    assert POLICY["notebook_violations"](document)
    assert POLICY["notebook_violations"](cleaned) == []
    assert [(c["cell_type"], c["source"]) for c in cleaned["cells"]] == [(c["cell_type"], c["source"]) for c in document["cells"]]
    assert cleaned["metadata"]["kernelspec"] == document["metadata"]["kernelspec"]
    assert cleaned["metadata"]["language_info"] == document["metadata"]["language_info"]
    assert all(c["metadata"]["tags"] == ["keep"] for c in cleaned["cells"])
    assert document["cells"][1]["outputs"]  # Cleanup does not mutate the source object.


def test_empty_or_malformed_json_cannot_pass_as_an_output_free_notebook():
    for malformed in ({}, [], {"nbformat": 4, "nbformat_minor": 5, "cells": {}, "metadata": {}},
                      {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
                       "cells": [{"cell_type": "code", "source": [3], "metadata": {}, "outputs": {}}]}):
        assert POLICY["notebook_violations"](malformed)


def test_staged_outputs_are_rejected_even_when_the_working_notebook_is_clean(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], capture_output=True, check=True)
    path = tmp_path / "example.ipynb"
    document = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
                "cells": [{"cell_type": "code", "source": ["print(1)"], "metadata": {},
                           "execution_count": 1, "outputs": [{"text": ["toy"]}]}]}
    path.write_text(json.dumps(document), encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "example.ipynb"], capture_output=True, check=True)
    path.write_text(json.dumps(POLICY["clean_notebook"](document)), encoding="utf-8")
    issues = POLICY["repository_violations"](tmp_path)
    assert any("[index]:" in issue and "saved outputs" in issue for issue in issues)
    assert not any("[working tree]:" in issue for issue in issues)
    subprocess.run(["git", "-C", str(tmp_path), "add", "example.ipynb"], capture_output=True, check=True)
    assert POLICY["repository_violations"](tmp_path) == []
