"""데모 요인 패널 — 결정성·형식·학습 결과(활성화 조건·부호 서사)를 고정한다."""
from __future__ import annotations

import math

import pytest

from src.core.seed import get_seed
from src.modelops import datasets, factor_panel

# 생성 규칙이 바뀌면 이 값과 아래 부호 단언을 함께 확인한다.
HASH_PREFIX = {"namwon": "c027e5128333", "shinan": "f0180b517d0a"}
STRUCTURE = ["youth_ratio", "aging_index", "working_age_pop", "elderly_pop", "death_cancellations"]
FACILITY = ["edu_facility_density", "medical_facility_density", "child_welfare_density", "facility_capacity"]
COEF_SIGN = {"youth_ratio": 1, "aging_index": -1, "working_age_pop": 1, "elderly_pop": -1, "death_cancellations": -1,
             "edu_facility_density": 1, "medical_facility_density": 1, "child_welfare_density": 1, "facility_capacity": 1}

REGIONS = get_seed()["regions"]


@pytest.mark.parametrize("region", REGIONS, ids=lambda r: r["id"])
def test_panel_is_deterministic_and_pinned(region):
    rows = factor_panel.generate(region)
    assert rows == factor_panel.generate(region)
    assert datasets._hash(rows)[:12] == HASH_PREFIX[region["id"]]


@pytest.mark.parametrize("region", REGIONS, ids=lambda r: r["id"])
def test_panel_shape_and_derived_relations(region):
    rows = factor_panel.generate(region)
    assert [row["base_ym"] for row in rows] == factor_panel.MONTHS
    assert len(rows) == 120 and rows[0]["base_ym"] == 201501 and rows[-1]["base_ym"] == 202412
    for row in rows:
        values = [row[name] for name in [*factor_panel.FEATURES, "y"]]
        assert all(math.isfinite(v) for v in values)
        assert all(row[name] > 0 for name in [*factor_panel.FEATURES, "y"] if name != "death_cancellations")
        assert row["death_cancellations"] >= 0
        youth = row["elderly_pop"] / row["aging_index"] * 100
        population = youth + row["elderly_pop"] + row["working_age_pop"]
        assert row["youth_ratio"] == pytest.approx(youth / population * 100, rel=2e-3)


@pytest.mark.parametrize("region", REGIONS, ids=lambda r: r["id"])
def test_training_beats_baseline_and_keeps_narrative_signs(region):
    rows = factor_panel.generate(region)
    artifact, metrics, baseline = factor_panel.train(rows)
    assert baseline["name"] == "same_month_last_year"
    assert metrics["mae"] < baseline["mae"]  # 활성화 조건(MAE < 기준선)
    assert artifact["eval"]["period"] == {"from": 202401, "to": 202412, "n": 12}
    assert artifact["features"] == factor_panel.FEATURES
    coef = dict(zip(artifact["features"], artifact["coef"]))
    assert {name: math.copysign(1, value) for name, value in coef.items()} == COEF_SIGN
    last = rows[-1]
    contribution = {name: c * (last[name] - mean) / std for name, c, mean, std
                    in zip(artifact["features"], artifact["coef"], artifact["train_means"], artifact["train_stds"])}
    assert all(contribution[name] < 0 for name in STRUCTURE)
    assert all(contribution[name] > 0 for name in FACILITY)


def test_available_mock_lists_panel_after_existing_sources():
    rows = datasets.available("mock")
    ids = [row["id"] for row in rows]
    assert ids[:len(REGIONS)] == ["mock_" + r["id"] for r in REGIONS]
    panels = [row for row in rows if row["id"].startswith("mock_factor_")]
    assert [row["region_id"] for row in panels] == [r["id"] for r in REGIONS]
    assert all(row["target"] == "living_population" and row["observed_from"] == 201501 and row["observed_to"] == 202412 for row in panels)
