"""prepared/README.md 생성: 공통 규칙 + MANIFEST 기반 데이터셋·시트별 노트."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import PREPARED_ROOT

_HEADER = """# platform/data/prepared — 2단계 전처리 산출물

extracted/ 의 시트별 CSV(41개)를 시트당 1개 prepared CSV 로 변환한 결과다.
시트 하나 = CSV 하나 = 향후 DB 테이블 하나. **파일 간 조인, 코드 체계 통일,
연령 구간 변환, 값 라벨 통일은 하지 않았다** — 이는 3단계(DDL/적재) 몫이다.

## 폴더 규약

```
prepared/<dataset>/<NN>_<slug>.csv          # 시트별 데이터/설명 CSV
prepared/<dataset>/<NN>_<slug>.schema.json  # 데이터 시트만. 컬럼 매핑·형·행수·드롭 근거
prepared/MANIFEST.csv                       # 41행. 시트별 처리 요약
prepared/README.md                          # 이 문서
```

## 공통 규칙 요약

- 설명 시트(kind=meta) 2개는 바이트 그대로 복사했다: `namwon_bccard_consumption/01_data_layout.csv`,
  `nowon_kt_festival/01_overview.csv`. schema.json 은 없다.
- 데이터 시트는 헤더 정확히 1행, 컬럼명은 영문 snake_case(원본 한글명은 schema.json 의
  `source_name` 참조), 코드류 컬럼은 문자열, 기준년월/월별은 YYYYMM 정수(`yearmonth`),
  금액·인원은 원값(반올림 없음), 비율 컬럼은 0~1 소수 + `_ratio` 접미.
- 합계·소계·평균 행은 제거하고 제거 행수를 schema.json `dropped_rows` 에 기록했다(재계산 가능).
- 결측은 빈 문자열로 두었다(0으로 채우지 않음). 격자 결손은 채우지 않고 기록만 한다.
- 와이드 표(연도×월, (지역명,값) 쌍 반복, 연도 블록 세로 반복, 좌우 병렬 기간)는 시트
  안에서 롱 포맷으로 언피벗했다 — 파일 간 통합이 아니라 시트 1개 안의 변환이다.
- 인코딩 UTF-8(BOM 없음), `newline=""`, `QUOTE_MINIMAL`.
- 각 데이터 시트의 원본 컬럼 대응·타입·드롭 근거·언피벗 방식은 같은 폴더의
  `<NN>_<slug>.schema.json` 을 우선 참조한다. 아래 노트는 schema.json 에 없는
  맥락(중복 블록, 부속표 제외 사유, uncertain 판단 근거)만 요약한다.
"""

_DATASET_TITLES = {
    "namwon_bccard_consumption": "A. namwon_bccard_consumption (BC카드 남원 소비, 2개)",
    "namwon_kt_visitors": "B. namwon_kt_visitors (KT 남원 방문객, 2개)",
    "nowon_kt_festival": "C. nowon_kt_festival (KT 노원 축제, 11개: meta 1 + data 10)",
    "gwto_kt_tourism_indicators": "D. gwto_kt_tourism_indicators (KT 강원 관광지표, 26개)",
}


def _load_schema_notes(out_csv: str) -> list[str]:
    schema_path = (PREPARED_ROOT / out_csv).with_suffix("").with_suffix(".schema.json")
    if not schema_path.exists():
        return []
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return data.get("notes", [])


def write_readme(manifest_rows: list[dict[str, Any]]) -> Path:
    lines = [_HEADER]

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, title in _DATASET_TITLES.items():
        rows = by_dataset.get(dataset, [])
        if not rows:
            continue
        lines.append(f"\n## {title}\n")
        for row in rows:
            slug = Path(row["out_csv"]).stem
            lines.append(
                f"### `{slug}` ({row['kind']}, rows_out={row['rows_out']},"
                f" dropped={row['rows_dropped']})"
            )
            if row["kind"] == "meta":
                lines.append("- 설명 시트, 바이트 그대로 복사.\n")
                continue
            notes = _load_schema_notes(row["out_csv"])
            if notes:
                for n in notes:
                    lines.append(f"- {n}")
            else:
                lines.append("- 특기 사항 없음(원본 그대로 컬럼명/형만 정리).")
            lines.append("")

    path = PREPARED_ROOT / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
