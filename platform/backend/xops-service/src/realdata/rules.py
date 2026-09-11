"""R5 진단 규칙 — 임계치·대응 후보 매핑을 데이터로 정의한다(계약 §5, rules_v1).

효과 수치·인구 증가 효과·RICE·policyStrategies는 이 모듈에 두지 않는다(계약 금지) — 여기 값은
규칙 임계치·문구뿐이고 실제 판정·근거 계산은 analysis.py가 한다.
"""

from __future__ import annotations

RULES_VERSION = "rules_v1"

# 진단 공통 게이트 — 관측 월 수가 이보다 적으면 해당 타깃의 규칙을 평가하지 않는다(근거 부족).
MIN_OBSERVED_MONTHS = 12
YOY_DECLINE_STREAK_MONTHS = 3
MIN_COMMON_MONTHS_FOR_DIVERGENCE = 3
DIVERGENCE_RATIO_DEVIATION_THRESHOLD = 0.20
SEASONAL_DEVIATION_THRESHOLD = 0.20

TARGET_LABELS = {
    "nonlocal_visitors": "외지인 방문객",
    "observed_sales_krw": "관측 소비",
}

_LIMITATION_SALES_K = "BC카드 소비는 k≥3 비식별 조건에 따라 일부 거래(셀)가 통계에서 제외될 수 있습니다."
_LIMITATION_SALES_GAP = "관측 소비는 2020~2021년이 공백입니다(2019·2022·2023만 관측)."
_LIMITATION_VISITORS_CAP = "KT 방문객 관측은 202310까지입니다(그 이후는 관측되지 않습니다)."

LIMITATIONS: dict[str, list[str]] = {
    "nonlocal_visitors": [_LIMITATION_VISITORS_CAP],
    "observed_sales_krw": [_LIMITATION_SALES_K, _LIMITATION_SALES_GAP],
    "both": [_LIMITATION_VISITORS_CAP, _LIMITATION_SALES_K, _LIMITATION_SALES_GAP],
}


def limitations_for(target: str) -> list[str]:
    return list(LIMITATIONS.get(target, []))


# 진단 rule_id → 대응 후보(카테고리·문구). 정책 효과 수치·인구 증가 효과는 담지 않는다(계약 R5).
RESPONSE_CANDIDATES: dict[str, dict[str, str]] = {
    "yoy_decline_streak_nonlocal_visitors": {
        "category": "방문 유인",
        "text": "외지인 방문객이 최근 3개월 연속 전년동월 대비 감소했습니다. 방문 유인 프로그램(관광 연계 이벤트 등) 검토를 제안합니다.",
    },
    "yoy_decline_streak_observed_sales_krw": {
        "category": "소비 유인",
        "text": "관측 소비가 최근 3개월 연속 전년동월 대비 감소했습니다. 지역 소비 촉진 방안 검토를 제안합니다.",
    },
    "visit_sales_divergence": {
        "category": "체류 소비 연계",
        "text": "방문객 대비 관측 소비 비율이 과거 평균과 괴리되었습니다. 체류 소비 연계(체류시간 확대, 소비처 접근성 등) 점검을 제안합니다.",
    },
    "seasonal_deviation_nonlocal_visitors": {
        "category": "계절 대응",
        "text": "이번 달 방문객이 과거 같은 달 평균 대비 크게 벗어났습니다. 계절 요인 대응(시기별 프로그램 조정) 검토를 제안합니다.",
    },
    "seasonal_deviation_observed_sales_krw": {
        "category": "계절 대응",
        "text": "이번 달 관측 소비가 과거 같은 달 평균 대비 크게 벗어났습니다. 계절 요인 대응 검토를 제안합니다.",
    },
}
