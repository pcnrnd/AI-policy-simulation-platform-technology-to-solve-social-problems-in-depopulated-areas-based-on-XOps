# Repository data policy

Original datasets, analysis report packages, research notes, local review state,
DVC working state and service checklists are kept outside Git tracking. Files
already present in `platform/data/real/` remain on disk for the existing loaders;
a fresh clone does not contain them. Obtain required originals from the approved
project custodian or verified private backup, retain their recorded hashes, and
place them in that local directory before running the existing load scripts.
Do not infer that a Git clone or bundle contains external XLSX/DVC payloads.

The local review configuration stays in ignored `.review/verification.yaml`;
its private migrated copy is managed in the review-system artifacts, outside the
repository. The technical service documentation and `docs/xops-openapi.json`
remain versioned API contracts. The demo seed, administrative name mapping and
design figures remain versioned.

Keep notebook source and kernel/language descriptions; clear saved outputs,
attachments, execution counts and execution metadata before staging notebooks.
Run `python tools/repository_data_policy.py` to check all tracked notebooks and
private paths, including both working files and the staged Git index. Stage
cleaned notebooks before running this check. Basic notebook structure is also
validated so malformed JSON cannot pass as an empty notebook. The backend pytest
suite includes the same check and verifies
that ignore rules also protect newly created files. No extra dependencies or
pre-commit installation are required.

The four synthetic realdata test fixtures have an independent deterministic
generator and a byte-reproduction check described in
[`tests/fixtures/realdata/README.md`](../platform/backend/xops-service/tests/fixtures/realdata/README.md).
They contain invented observations and parameters, not source dataset rows.

This change prevents future tracking and sanitizes current notebook outputs.
Earlier Git history still contains original objects until the separate, approved
history-cleanup phase finishes; private originals and backups remain preserved.
