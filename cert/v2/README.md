# cert/v2 — 2026 공인인증 2건

이 폴더는 2026년 공인인증 대상 2건을 각각 독립 프로젝트로 둔다. 이전에는 두 앱이
`cert/v2/` 최상위 한 프로젝트(공용 `lib/`·`pyproject.toml`·`uv.lock`·`Dockerfile`)를
공유했으나, 인증 산출물은 각자 독립적으로 빌드·배포·검증 가능해야 하므로 분리했다.

| 인증 | 폴더 | 포트 | 차트 |
|---|---|---|---|
| 데이터 시각화 처리 응답속도 | [`chart_speed/`](chart_speed/) | 8505 | 5종 |
| 데이터 시각화 유형 | [`chart_type/`](chart_type/) | 8506 | 8종 |

("데이터 시각화 처리 응답속도"·"데이터 시각화 유형"은 앱 진입 파일의 제목 문자열을
그대로 옮긴 표기다. 정식 인증 명칭은 이 문서가 창작하지 않는다.)

## 이전 → 새 경로

| 이전 (분리 전, `cert/v2/`) | 새 위치 |
|---|---|
| `chart_speed.py` | `chart_speed/chart_speed.py` |
| `chart_type.py` | `chart_type/chart_type.py` |
| `lib/dashboard.py`, `lib/visualization.py` | 각 `chart_speed/lib/`, `chart_type/lib/`에 복사 (동일 파일 2벌) |
| `pyproject.toml` | 각 폴더에 복사, `name`만 `cert-chart-speed`/`cert-chart-type`로 변경 |
| `uv.lock` | 각 폴더에 복사 후 재생성(루트 패키지명만 반영, 의존성 126개 핀 무변경) |
| `Dockerfile` | 각 폴더로, `APP`/`PORT` 분기 제거·앱별 포트 고정 |
| `.dockerignore` | 각 폴더에 동일 내용 복사 |
| `data_2026/`(7파일) | 각 폴더에 복사(원본과 sha256 동일 확인, 재생성 안 함) |
| `measure_speed.py` | `chart_speed/tools/measure_speed.py` (CLI·계측 로직 무변경) |
| `prepro_2026.py` | `tools/prepro_2026.py` (`--input`/`--output-dir` 필수 인자로 변경, 하드코딩 경로 제거) |
| `data_check_2026.ipynb` | `tools/data_check_2026.ipynb` (DB 경로 셀만 `chart_type/data_2026/database.db` 기준으로 수정) |
| `cert-compose.yml` | 그대로(최상위), `build`/`volumes`만 각 폴더로 변경 |

## 공통 조건

- 데이터: 남원시 일반음식점 2021~2026년(`prepro_2026.py` 산출물), 두 앱 각자 독립 사본을 갖되 내용은 원본과 sha256 동일.
- 지연: `chart_speed`만 `CHART_SLEEP_MS`(기본 350ms) 적용, `chart_type`은 지연 없음.
- 실행: 각 폴더가 완전히 독립된 uv 프로젝트/Docker 이미지 — 어느 한쪽만 떼어내 빌드·기동 가능(교차 참조 없음).
- 상위 `cert-compose.yml`로 두 앱을 동시에 올릴 수 있다(`docker compose -f cert-compose.yml up -d --build`, 포트 8505/8506).

## 도구

- `tools/prepro_2026.py`: 원본 CSV → `data_2026/` 생성(각 앱 폴더를 `--output-dir`로 지정). 이번 분리 작업에서는 실행하지 않았다.
- `tools/data_check_2026.ipynb`: `chart_type/data_2026/database.db` 기준 데이터 점검 노트북(8종 차트 집계 쿼리 확인용).
- `chart_speed/tools/measure_speed.py`: Playwright 기반 응답속도 실측. `chart_type` 렌더 확인에도 그대로 쓸 수 있다(`--charts 8`).
