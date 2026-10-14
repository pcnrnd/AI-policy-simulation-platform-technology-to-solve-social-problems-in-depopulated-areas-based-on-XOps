These four fixtures are synthetic: `artifact_v1.json`, `artifact_v2.json`,
`bc_dong_industry_sales.json` and `kt_monthly_visitors.json`.
`generate_synthetic.py` invents every numeric observation, model parameter and
metric using fixed arithmetic; it reads no source data, database, clock or seed.
The region identifiers exercise the retained administrative mapping. The date
range supplies 16 months for lag/training tests; sales intentionally omit two
months to test missing observations without zero filling. Artifacts retain the
existing JSON schema and have two distinguishable synthetic versions.

Regenerate with `python tests/fixtures/realdata/generate_synthetic.py` from the
service directory; verify with the same command followed by `--check`.
The ordinary pytest suite also checks reproducibility and these test contracts.
`bc_dong_industry_sales_unmapped.json` remains the fictitious-name rejection case.
