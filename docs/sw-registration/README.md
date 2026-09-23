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

코드 기준 경계이며 `tests/integration/test_sw_entrypoints.py`가 import 방향을 검사한다.

| 구분 | SW-01 (DataOps) | SW-02 (MLOps) |
|---|---|---|
| API | `/api/v3/dataops/*` | `/api/v3/monitoring/*`, `/api/v3/orchestration/*` |
| 고유 모듈 | `src/dataops/{catalog,adapters,backends,query_builder,safety,results,service,liveness}.py`, `src/schemas/dataops.py` | `src/mlops/{monitoring,orchestration,training}/*`, `src/schemas/{monitoring,orchestration}.py` |
| 단독 진입점 | `sw_main:create_sw01_app` | `sw_main:create_sw02_app` |

**공유 기반(두 SW 공통, 어느 한쪽 고유 기능이 아님)** — `src/core/*`(설정·SQLite·시드·로그·예외), `src/auth/jwt.py`, `src/api/dependencies.py`.

**SW 간 의존 1건** — SW-02 → `src/mlops/monitoring/realdata_bridge.py` → `src/realdata/{candidates,jobs,monitoring,snapshot}.py` → `src/realdata/pg_reader.py` → `src/dataops/safety.py`(`assert_safe_identifier`, 식별자 검증 함수 1개). SW-02가 SW-01의 CRUD·카탈로그 경로를 쓰는 것은 아니다. 반대 방향(SW-01 → SW-02)은 없다.

**두 SW 모두에서 제외** — `src/api/v3/overview.py`와 `src/dataops/summary.py`(Overview 대시보드 롤업, `mlops.registry`를 읽음), `/api/v3/realdata/*` 라우터 4종, `report-service`·`dashboard-service`, 프런트엔드 화면.

**`src/realdata/` 포함 여부 (미결)** — 2026-09-11 초안은 독립 검토 대기를 이유로 SW-02 범위에서 뺐다. 그 뒤 `6bb75bd`에서 SW-02 라우터가 브리지를 통해 위 realdata 모듈을 import하므로, 소스코드 제출 시 이 모듈들이 없으면 SW-02가 기동되지 않는다. 설명자료에는 "의존 모듈로 동반 제출, 기능 서술은 브리지 경유 조회만"으로 적었다. 확정은 루트 결정 사항이다.

## 전체 플랫폼 실행과의 관계

`main.py`(`main:app`)는 위 전부와 제외 모듈을 한 프로세스로 띄우는 기존 진입점이며 Docker Compose도 이것을 쓴다. `sw_main.py`는 SW 등록 시연용으로 라우터만 골라 띄우며 `main.py`·Compose 동작을 바꾸지 않는다.
