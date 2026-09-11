"""rules.py는 데이터(임계치·대응 문구)만 담는다 — 효과 수치·RICE류 키가 없어야 한다(계약 R5 금지)."""

from __future__ import annotations

from src.realdata import rules

_FORBIDDEN_TOKENS = ("rice", "budget", "effect", "population_gain", "score")


def test_rules_version_is_v1() -> None:
    assert rules.RULES_VERSION == "rules_v1"


def test_response_candidates_cover_every_rule_id_used_by_analysis() -> None:
    expected = {
        "yoy_decline_streak_nonlocal_visitors",
        "yoy_decline_streak_observed_sales_krw",
        "visit_sales_divergence",
        "seasonal_deviation_nonlocal_visitors",
        "seasonal_deviation_observed_sales_krw",
    }
    assert expected == set(rules.RESPONSE_CANDIDATES)


def test_response_candidates_have_category_and_text_only_no_effect_numbers() -> None:
    for rule_id, template in rules.RESPONSE_CANDIDATES.items():
        assert set(template) == {"category", "text"}, rule_id
        for key in template:
            assert not any(token in key.lower() for token in _FORBIDDEN_TOKENS)


def test_limitations_for_known_targets_are_non_empty() -> None:
    assert rules.limitations_for("nonlocal_visitors")
    assert rules.limitations_for("observed_sales_krw")
    assert rules.limitations_for("both")


def test_limitations_for_unknown_target_is_empty_not_error() -> None:
    assert rules.limitations_for("no-such-target") == []
