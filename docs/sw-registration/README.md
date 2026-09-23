# 2026 SW 등록 — 인구감소 과제 (SW-01·SW-02)

> 초안 (2026-09-23 개정, 미확정) — 제출·신청 확정본이 아님

설명자료 MD 원본은 이 폴더가 정본이다. 스테이징(`F:\rnd_work\2026-sw-td\인구감소\SW등록_기술문서\SW-0?\02_설명서\`)의 2026-09-11 초안(기준 `f81df7a`)을 옮겨 와 기준 커밋을 갱신했다.

| 문서번호 | 프로그램명 | 설명자료 | 실행 README |
|---|---|---|---|
| SW-01 | 메타데이터 기반 다기종 데이터 아카이빙 관리 프로그램 | [SW-01/…_설명자료.md](SW-01/SW-01_메타데이터기반_다기종데이터_아카이빙관리_설명자료.md) | [SW-01/README.md](SW-01/README.md) |
| SW-02 | 인공지능 모델 성능 모니터링 및 재학습 관리 프로그램 | [SW-02/…_설명자료.md](SW-02/SW-02_AI모델성능모니터링_재학습관리_설명자료.md) | [SW-02/README.md](SW-02/README.md) |

- 기준 커밋: `2e71da4224cc4a6970b6888d5e6968eb243dd7ae` (master). 이 커밋 이후 이 브랜치의 변경은 문서·단독 진입점(`sw_main.py`)과 그 테스트뿐이다.
- `f81df7a` → `2e71da4` 사이 백엔드 변경 중 SW 범위의 마지막 커밋은 `8d8e7b8`(모니터링 `window`)이다. `2e71da4`는 프런트 1파일만 바꿨다.

## 기능 경계

등록 범위는 루트 결정(2026-09-23)을 따른다. import 방향은 `tests/integration/test_sw_entrypoints.py`가 검사한다.

| 구분 | SW-01 (DataOps) | SW-02 (MLOps) |
|---|---|---|
| API | `/api/v3/dataops/*` | `/api/v3/monitoring/*`, `/api/v3/orchestration/*` |
| 등록 범위 파일 | `src/dataops/*`, `src/api/v3/dataops.py`, `src/schemas/dataops.py`, `platform/scripts/load_real_data.py`, `platform/scripts/load_external_data.py` | `src/mlops/{monitoring,orchestration,training}/*`(`realdata_bridge`·`sources` 포함), `src/api/v3/{monitoring,orchestration}.py`, `src/schemas/{monitoring,orchestration}.py` |
| 단독 진입점 | `sw_main:create_sw01_app` | `sw_main:create_sw02_app` |

파일별 목록은 각 설명자료 §6에 있다. `sw_main.py`는 두 SW가 함께 쓰는 시연용 진입점이다.

**등록 범위 밖 공용 의존** — 두 SW가 기동에 필요로 하지만 어느 SW의 등록 범위에도 넣지 않는다.

| 모듈 | 쓰는 쪽 | 용도 |
|---|---|---|
| `src/core/*` | 양쪽 | 설정·SQLite(`db.py`)·시드·로그·예외 |
| `src/auth/jwt.py`, `src/api/dependencies.py` | 양쪽 | JWT 발급·검증, scope 검사 |
| `src/realdata/{candidates,jobs,monitoring,snapshot,pg_reader,errors,models,evaluate,features}.py`, `namwon_dong_map.json` | SW-02(`realdata_bridge` 경유, 뒤 3개는 호출 시점 import) | 실데이터 모델 평가·학습 job 조회. 이 모듈의 기능은 SW-02 기능으로 서술하지 않는다 |

**하나의 FastAPI 프로세스 공유** — 통합 플랫폼(`main:app`, Compose)은 두 SW와 Overview·realdata 라우터를 한 프로세스로 띄우고 `init_db()`로 같은 SQLite를 초기화한다. 단독 진입점을 따로 띄워도 같은 `XOPS_DB_PATH`를 가리키면 SQLite 파일을 공유한다.

**SW 간 교차 import** (반대 방향 SW-01 → SW-02 기능 경로는 없음)

- SW-02 → `realdata/pg_reader.py` → `src/dataops/safety.py`(`assert_safe_identifier`, 식별자 검증 함수 1개). SW-01의 카탈로그·CRUD 경로는 쓰지 않는다.
- `src/dataops/summary.py`(SW-01 등록 범위 안)는 Overview 대시보드 롤업 전용으로 `src.mlops.orchestration.registry`를 읽는다. `/api/v3/overview`에서만 쓰이며 SW-01 라우터·단독 진입점은 이 파일을 import하지 않는다.

**두 SW 모두에서 제외** — `src/api/v3/overview.py`, `/api/v3/realdata/*` 라우터 4종과 `src/realdata/{analysis,rules}.py`, `src/schemas/realdata.py`, `report-service`·`dashboard-service`, 프런트엔드 화면.

## 전체 플랫폼 실행과의 관계

`main.py`(`main:app`)는 위 전부와 제외 모듈을 한 프로세스로 띄우는 기존 진입점이며 Docker Compose도 이것을 쓴다. `sw_main.py`는 SW 등록 시연용으로 라우터만 골라 띄우며 `main.py`·Compose 동작을 바꾸지 않는다.
