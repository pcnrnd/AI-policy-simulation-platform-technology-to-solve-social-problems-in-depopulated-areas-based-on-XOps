"""외부데이터 2단계 전처리: extracted/ 시트별 CSV → prepared/ 시트별 CSV(시트당 1파일, 통합 없음).

시트 하나 = CSV 하나 = 향후 DB 테이블 하나. 파일 간 조인·코드 체계 통일·연령 구간
변환·값 라벨 통일은 하지 않는다(3단계 몫). 상세 규칙은 데이터셋별 모듈
(prepare_rules/*.py)에 둔다.

사용법:
    python platform/scripts/prepare_sheets.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prepare_rules import bccard, common, festival, gwto, kt_visitors, readme

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("prepare_sheets")

_BUILDERS = {
    "namwon_bccard_consumption": bccard.build,
    "namwon_kt_visitors": kt_visitors.build,
    "nowon_kt_festival": festival.build,
    "gwto_kt_tourism_indicators": gwto.build,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    manifest_rows = []
    for dataset, builder in _BUILDERS.items():
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
