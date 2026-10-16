"""데모 요인분석용 합성 패널 — 인구 구조·시설 지표 9개와 월 생활인구(2015-01~2024-12, 지역별 120행).

변수 집합은 2025 연차보고서의 요인 패널(유소년 비율·노령화지수·생산연령인구·고령인구·사망말소·
교육/의료/아동복지 시설 밀도·시설 수용력)을 따른다. 실제 관측이 아니라 데모 표시용 합성값이다.

- 결정적: 지역마다 문자열 시드 난수(`random.Random`, PYTHONHASHSEED와 무관)로 만든다. 같은 seed 지역이면
  언제나 같은 행이 나오고, 등록 시 내용 해시로 확인한다.
- 주민등록 인구는 seed `history`(연 10점)를 월로 보간해 쓴다 — 데모 화면의 인구값과 맞춘다.
- 생활인구 = 기준값 × (1 + 계절 1% + Σ 영향 계수 × 표준화 요인) + 소음. 시설 밀도는 인구가 줄면서
  올라가므로 최근값이 평균 위(기여 +), 인구 구조 요인은 최근값이 불리한 쪽(기여 −)이 되도록 잡았다.
  파라미터를 바꾸면 tests/unit/test_factor_panel.py의 부호·해시 단언이 함께 바뀌어야 한다.
"""
from __future__ import annotations

import math
import random
from typing import Any

from src.mlops.training.model import RidgeRegressor
from src.realdata.evaluate import metric_bundle

FEATURES = ["youth_ratio", "aging_index", "working_age_pop", "elderly_pop", "death_cancellations",
            "edu_facility_density", "medical_facility_density", "child_welfare_density", "facility_capacity"]
TARGET = "living_population"
SOURCE_PREFIX = "mock_factor_"
MONTHS = [year * 100 + month for year in range(2015, 2025) for month in range(1, 13)]
EVAL_MONTHS = 12  # 마지막 12개월(2024년)로 검증, 기준선은 전년 같은 달 관측값
BASELINE_NAME = "same_month_last_year"

# 표준화 요인 1단위가 생활인구 기준값에 주는 영향(비율 × 0.1). 부호가 화면 서사의 방향이다.
_EFFECTS = {"youth_ratio": 0.18, "aging_index": -0.20, "working_age_pop": 0.14, "elderly_pop": -0.07,
            "death_cancellations": -0.06, "edu_facility_density": 0.12, "medical_facility_density": 0.10,
            "child_welfare_density": 0.05, "facility_capacity": 0.07}
# yr·es: 유소년·고령 비중(%) 시작→끝, death: 월 사망말소 시작→끝, 시설 수 초기값, living: 생활인구 기준값
_PARAMS = {
    "namwon": {"yr": (11.2, 8.4), "es": (22.0, 31.0), "death": (55, 72), "edu": 96, "med": 140, "child": 24, "cap": 3200, "living": 240000},
    "shinan": {"yr": (8.6, 6.1), "es": (31.0, 41.0), "death": (38, 50), "edu": 58, "med": 62, "child": 11, "cap": 1500, "living": 160000},
}


def _params(region: dict[str, Any]) -> dict[str, Any]:
    # ponytail: 파라미터가 없는 지역은 남원 값을 인구 규모 비율로 쓴다 — 지역별 표가 필요해지면 _PARAMS에 추가.
    if region["id"] in _PARAMS:
        return _PARAMS[region["id"]]
    base, scale = _PARAMS["namwon"], region["history"][-1] / 77200
    return {**base, "death": tuple(v * scale for v in base["death"]),
            **{key: round(base[key] * scale) for key in ("edu", "med", "child", "cap", "living")}}


def _population(history: list[float], index: int) -> float:
    """연 단위 history(각 해 중순 값)를 월 index(0..119)로 선형 보간한다. 양 끝은 고정."""
    position = index / 12.0 - 0.5
    if position <= 0:
        return float(history[0])
    low = min(len(history) - 2, int(math.floor(position)))
    fraction = min(1.0, position - low)
    return history[low] + (history[low + 1] - history[low]) * fraction


def generate(region: dict[str, Any]) -> list[dict[str, Any]]:
    """지역 하나의 월 패널 120행. 같은 입력이면 언제나 같은 결과다."""
    p = _params(region)
    rng = random.Random("factor-panel-v1:" + region["id"])
    noise = {"yr": 0.0, "es": 0.0, "pop": 0.0}
    edu, med, child, cap = p["edu"], p["med"], p["child"], p["cap"]
    rows: list[dict[str, Any]] = []
    for index, month in enumerate(MONTHS):
        t = index / (len(MONTHS) - 1)
        for key in noise:
            noise[key] = 0.7 * noise[key] + rng.gauss(0, 1)
        youth_ratio = p["yr"][0] + (p["yr"][1] - p["yr"][0]) * t + 0.45 * noise["yr"]
        elderly_share = p["es"][0] + (p["es"][1] - p["es"][0]) * t + 0.9 * noise["es"]
        if rng.random() < 0.04:
            edu += rng.choice([-1, -1, 1])
        if rng.random() < 0.08:
            med += rng.choice([-2, 2, 3])
        if rng.random() < 0.05:
            child += rng.choice([-1, 1, 1])
        if rng.random() < 0.04:
            cap += rng.choice([40, 60, 80])
        population = _population(region["history"], index) * (1 + 0.006 * noise["pop"])
        winter = 1.12 if month % 100 in (12, 1, 2) else 1.0
        youth, elderly = population * youth_ratio / 100, population * elderly_share / 100
        deaths = (p["death"][0] + (p["death"][1] - p["death"][0]) * t) * winter + rng.gauss(0, 2)
        rows.append({"base_ym": month, "youth_ratio": round(youth_ratio, 3), "aging_index": round(elderly / youth * 100, 2),
                     "working_age_pop": round(population - youth - elderly), "elderly_pop": round(elderly),
                     "death_cancellations": max(0, round(deaths)),
                     "edu_facility_density": round(edu / population * 1e4, 3), "medical_facility_density": round(med / population * 1e4, 3),
                     "child_welfare_density": round(child / population * 1e4, 3), "facility_capacity": round(cap / population * 1e3, 2)})
    count = len(rows)
    means = {name: sum(row[name] for row in rows) / count for name in FEATURES}
    stds = {name: math.sqrt(sum((row[name] - means[name]) ** 2 for row in rows) / count) for name in FEATURES}
    for row in rows:
        season = 0.01 * math.sin(2 * math.pi * (row["base_ym"] % 100 - 4) / 12)
        effect = sum(_EFFECTS[name] * 0.1 * (row[name] - means[name]) / stds[name] for name in FEATURES)
        row["y"] = round(p["living"] * (1 + season + effect) + rng.gauss(0, p["living"] * 0.004))
    return rows


def source(region: dict[str, Any], content_hash: str) -> dict[str, Any]:
    """DataOps 데모 원천 목록의 한 행(등록 전 원본)."""
    return {"id": SOURCE_PREFIX + region["id"], "name": region["name"] + " 인구 구조·시설 지표 (생활인구)",
            "region_id": region["id"], "target": TARGET, "source": "mock", "content_hash": content_hash,
            "observed_from": MONTHS[0], "observed_to": MONTHS[-1]}


def _matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(row[name]) for name in FEATURES] for row in rows]


def train(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """앞 108개월로 학습해 마지막 12개월을 검증하고(기준선 = 전년 같은 달), 전체로 다시 학습한 계수를 저장한다."""
    split = len(rows) - EVAL_MONTHS
    targets = [float(row["y"]) for row in rows]
    holdout = RidgeRegressor.fit(_matrix(rows[:split]), targets[:split], ridge_lambda=1.0, feature_names=FEATURES)
    actual = targets[split:]
    metrics = metric_bundle(actual, holdout.predict(_matrix(rows[split:])))
    baseline = {"name": BASELINE_NAME, **metric_bundle(actual, targets[split - 12:len(rows) - 12])}
    fit = RidgeRegressor.fit(_matrix(rows), targets, ridge_lambda=1.0, feature_names=FEATURES).to_dict()
    artifact = {"features": list(FEATURES), "coef": fit["coefficients"], "intercept": fit["intercept"],
                "train_means": fit["means"], "train_stds": fit["stds"], "lambda": fit["ridge_lambda"],
                "eval": {"period": {"from": rows[split]["base_ym"], "to": rows[-1]["base_ym"], "n": EVAL_MONTHS}}}
    return artifact, metrics, baseline
