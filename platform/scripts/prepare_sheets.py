"""외부데이터 2단계 전처리: extracted/ 시트별 CSV → prepared/ 시트별 CSV(시트당 1파일, 통합 없음).

시트 하나 = CSV 하나 = 향후 DB 테이블 하나. 파일 간 조인·코드 체계 통일·연령 구간
변환·값 라벨 통일은 하지 않는다(3단계 몫). 상세 규칙은 데이터셋별 모듈
(prepare_rules/*.py)에 둔다.

사용법:
    python platform/scripts/prepare_sheets.py                      # 전부
    python platform/scripts/prepare_sheets.py --only kto_spot_visitors   # 일부만(나머지 MANIFEST 행 유지)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from prepare_rules import bccard, common, festival, gwto, kt_visitors, kto_spot_visitors, mois_living_population, readme

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("prepare_sheets")

_BUILDERS = {
    "namwon_bccard_consumption": bccard.build,
    "namwon_kt_visitors": kt_visitors.build,
    "nowon_kt_festival": festival.build,
    "gwto_kt_tourism_indicators": gwto.build,
    "kto_spot_visitors": kto_spot_visitors.build,
    "mois_living_population": mois_living_population.build,
}


def kept_manifest_rows(only: set[str]) -> list[dict[str, str]]:
    """--only 실행 때 다시 만들지 않는 데이터셋의 기존 MANIFEST 행."""
    path = common.PREPARED_ROOT / "MANIFEST.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["dataset"] not in only]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=sorted(_BUILDERS), help="이 데이터셋만 다시 만든다(반복 가능)")
    args = parser.parse_args()

    only = set(args.only or [])
    manifest_rows = kept_manifest_rows(only) if only else []
    for dataset, builder in _BUILDERS.items():
        if only and dataset not in only:
            continue
        extracted_dir = common.EXTRACTED_ROOT / dataset
        _log.info("처리 시작: dataset=%s", dataset)
        rows = builder(extracted_dir)
        _log.info("처리 완료: dataset=%s 시트 %d개", dataset, len(rows))
        manifest_rows.extend(rows)

    manifest_path = common.write_manifest(manifest_rows)
    _log.info("MANIFEST 작성: %s (%d행)", manifest_path, len(manifest_rows))

    readme_path = readme.write_readme(manifest_rows)
    _log.info("README 작성: %s", readme_path)


if __name__ == "__main__":
    main()
