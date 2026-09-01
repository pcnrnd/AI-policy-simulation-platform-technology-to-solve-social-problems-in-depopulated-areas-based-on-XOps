# xops-service 기술문서

> 인구감소 R&D 플랫폼 — DataOps + MLOps 운영 평면 백엔드
> 경로: `platform/backend/xops-service/` · FastAPI (Python 3.10+) · 최종 갱신 2026-09-01

---

## 1. 개요

`xops-service`는 개발내용의 **DataOps(데이터 라이프사이클·아카이빙·Data API 빌더)** 와 **MLOps(모델 성능 모니터링·오케스트레이션)** 를 담당하는 단일 FastAPI 백엔드다. `platform/frontend`(React+Vite) 대시보드가 이 서비스의 `/api/v3` API를 호출해 동작한다.

### 서비스 분리 원칙
`platform/backend/`는 세 서비스로 나뉜다 — **xops-service(운영 평면)**, report-service(문서), dashboard-service(조회). DataOps와 MLOps는 **하나의 xops-service 안에서 내부 라우터/모듈로만 분리**한다. 두 도메인이 메타데이터·JWT scope·DB Adapter를 공유하고, 오케스트레이션(재학습)이 DataOps(데이터 준비) 위에 서기 때문이다. 더 잘게 쪼개는 것은 R&D 규모에서 과설계다.

### 담당 범위
- **포함**: DataOps 전체, MLOps 모니터링·오케스트레이션
- **미포함(후속)**: 시뮬레이션/공간정보 → dashboard-service, 리포팅 → report-service

---

## 2. 아키텍처

단일 FastAPI 앱 + `api/v3/router.py` 집약 + 도메인 패키지. `/api/v3` prefix와 `dataops_version: "3.0.0-R3"`는 프론트 계약에 맞춘 값이다.

```
platform/backend/xops-service/
├── main.py                     # create_app(): CORS·예외핸들러·init_db·/api/v3 마운트, GET / 헬스
├── pyproject.toml              # 의존성·pytest(80% 게이트)·ruff·mypy
├── requirements.txt            # Docker/CI 런타임 의존성(pyproject와 동기화)
├── Dockerfile · .dockerignore  # python:3.11-slim + uvicorn 8000
├── src/
│   ├── core/
│   │   ├── settings.py         # pydantic-settings — 임계치·경로·인증·CORS 전부 XOPS_ 환경변수
│   │   ├── db.py               # SQLite(WAL) 영속화 — 사용자소스·모델버전·실행이력
│   │   ├── seed.py             # mock_data.json 시드 로더(카탈로그·시계열)
│   │   ├── logger.py           # 구조화 JSON 로깅(print 금지)
│   │   └── exceptions.py       # 도메인 예외 계층 + FastAPI 핸들러
│   ├── auth/jwt.py             # HS256 발급/검증(stdlib), OAuth2 grant
│   ├── api/
│   │   ├── dependencies.py     # require_auth(scope) · require_client(prod 게이트)
│   │   └── v3/
│   │       ├── router.py       # dataops·monitoring·orchestration 집약
│   │       ├── dataops.py
│   │       ├── monitoring.py
│   │       └── orchestration.py
│   ├── schemas/                # Pydantic DTO (dataops·monitoring·orchestration)
│   ├── dataops/                # catalog·adapters·backends·query_builder·safety·service·results
│   └── mlops/
│       ├── monitoring/         # metrics·drift·outliers·explain
│       └── orchestration/      # events·evaluator·deployer·orchestrator·registry
└── tests/{unit,integration}/   # pytest, 커버리지 95%
```

**레이어 규약**: `from __future__ import annotations` + 전체 타입힌트, 함수 ≤50줄·파일 ≤300줄, 임계치 하드코딩 금지(전부 `Settings`), bare `except` 금지, 선택 ML 라이브러리 부재 시 graceful degradation.

---

## 3. 실행

### 로컬 (개발)
```powershell
cd platform/backend/xops-service
python -m pip install -e .[dev]           # 런타임+테스트 (,ml로 SHAP/numpy 추가)
python -m uvicorn main:app --reload   # uvicorn 기본 포트 8000
#  → http://localhost:8000/docs (Swagger), GET / 헬스체크
```
> 임포트가 `from src...`이므로 **cwd는 반드시 `xops-service/`**. 포트는 프레임워크 기본값 — 백엔드 uvicorn 8000, 프론트 vite 5173.

### Docker Compose (프론트 포함)
```powershell
docker compose -f platform/compose.xops.yaml up -d --build
#  → http://localhost:8080 (프론트, nginx가 /api를 xops-service:8000으로 프록시)
```

### 테스트
```powershell
python -m pytest tests/ --cov=src --cov-report=term-missing   # 80% 게이트
```

---

## 4. 도메인 상세

### 4.1 DataOps (`src/dataops/`)

**"저장소를 직접 노출하지 않고 메타데이터+API로 추상화"** 라는 명세를 구현. 요청 → 메타데이터 검색 → Adapter 선택 → 쿼리 생성 → 안전 검증 → 표준 REST 응답.

| 모듈 | 역할 |
|---|---|
| `catalog.py` | `MetadataCatalog` — 시드(mock_data.json 9종 = 합성 7 + 실데이터 2)는 읽기전용, 사용자 등록 소스는 SQLite. id·태그·객체명·설명 검색, add/remove |
| `adapters.py` | 표시용 Adapter 이름(프론트 계약) + 실행 어댑터 팩토리 `get_adapter`. DSN·드라이버가 없으면 `InMemoryAdapter` 로 degrade |
| `backends.py` | 실 저장소 실행 — psycopg(PostgreSQL·PostGIS·TimescaleDB)/pymongo. 읽기는 표시 SQL 그대로, 쓰기는 `build_write_sql` 파라미터 바인딩 문장 실행 |
| `results.py` | `ExecutionRequest`/`ExecutionResult` 계약 — `executed=False` 가 degrade 신호 |
| `query_builder.py` | 표준 SQL / MQL 생성 + 쓰기 실행용 `build_write_sql`(%s 바인딩)·`build_mongo_query`. 메서드·필터·정렬·페이징 + 메타데이터 적재범위(range) 자동 주입(SQL `BETWEEN` / MQL `$gte·$lte`) |
| `safety.py` | 인젝션 가드 — filter 단일 조건, 정렬·filter 컬럼 스키마 대조, 카탈로그 식별자·range 값 allowlist(`assert_safe_schema`), 쓰기 본문 스칼라 강제(`assert_safe_write_values`), 생성 SQL 주석/스택쿼리 차단 |
| `service.py` | `DataService` — 위 흐름을 엮어 `buildApiResponse` 계약(status·endpoint·adapter·archive_meta·range_scope·query_language·generated_query + GET 페이지네이션) 생성 |

- 모든 `/dataops/{source_id}` CRUD는 **JWT 필수**(GET=`data:read`, 쓰기=`data:write`).
- 저장소 유형에 따라 동일 요청이 SQL 또는 MQL로 변환됨(다기종). Mongo 소스는 `db.col.find({seq:{$gte,$lte}})` 형태 재현.
- **실 실행 조건**: 읽기는 DSN·드라이버가 있으면 항상 실 저장소를 탄다(`source_kind: "database"`,
  실패 시 스텁으로 degrade). 쓰기는 본문 값 있는 POST/PUT/PATCH·filter 있는 PUT/PATCH/DELETE 만
  실행하고 값은 전부 드라이버 파라미터로 바인딩한다 — 조건 미충족이면 종전 결정적 스텁 유지.
- **실데이터**: ds_08(전국 시군구 경계 251, PostGIS)·ds_09(남원·신안 사회복지시설 17, PostgreSQL)는
  `platform/data/real/` 원본을 `platform/scripts/load_real_data.py` 로 적재한 실제 공공데이터다.

### 4.2 MLOps 모니터링 (`src/mlops/monitoring/`)

지표·드리프트·이상치·설명가능성을 **실제 계산**(순수 Python, 외부 의존 없음).

| 모듈 | 내용 |
|---|---|
| `metrics.py` | `MetricCollector` — 6대 지표. 회귀 MSE·MAE, 분류 Accuracy·Precision·Recall·F1(혼동행렬 기반) |
| `drift.py` | `DriftDetector` — PSI = Σ(cur−ref)·ln(cur/ref), KL divergence. 임계 PSI≥0.2·KL≥0.1(설정값), epsilon clip |
| `outliers.py` | `OutlierDetector` — Z-score(기본), IQR. 경계·이상치 인덱스 반환 |
| `explain.py` | `ExplainabilityModule` — feature 중요도. SHAP 설치 시 사용, 미설치 시 평균 절대기여도 fallback |

- GET 계열(`/metrics`, `/drift`, `/explain`)은 대시보드용 시드 시계열 서빙, POST 계열은 입력으로 실계산.

### 4.3 MLOps 오케스트레이션 (`src/mlops/orchestration/`)

이벤트 기반 재학습 상태머신: `queued → preparing → training → evaluating → deploying → (succeeded | rolled_back)`, 승급 미달 시 `rejected`, 중복 트리거 시 `debounced`.

| 모듈 | 역할 |
|---|---|
| `events.py` | `RetrainEvent` + `EventBus` — model_id별 최소간격 debounce(`retrain_min_interval_minutes`), `manual`은 항상 통과 |
| `evaluator.py` | `Evaluator` — primary 지표 우선순위 **f1 > accuracy > mae > mse**로 후보 승급 판정 |
| `deployer.py` | `AutoDeployer` — canary → full. 후보 latency > 200ms(설정값)면 **자동 롤백**(직전 버전 유지, 트래픽 미전환) |
| `orchestrator.py` | `Orchestrator._handle_event` — 위 단계를 구동, `PipelineRun`(state·evaluation·deploy·active_version·candidate_metrics) 반환 |
| `registry.py` | `ModelRegistry` — 모델 스토어(3종 시드)+실행이력. 승급 성공 시 버전 갱신, SQLite 영속화. 모니터링·오케스트레이션 라우터 공유 싱글톤 |

> 인프로세스 구현. Airflow/Argo로 이관 시 각 stage를 operator로 매핑하면 된다.

---

## 5. API 레퍼런스 (`/api/v3`)

### DataOps
| 메서드 | 경로 | 설명 | 인증 |
|---|---|---|---|
| POST | `/dataops/token/{source_id}` | JWT 발급 | prod 게이트 |
| POST | `/dataops/oauth2/{source_id}` | OAuth2 Authorization Code Grant | prod 게이트 |
| GET | `/dataops/catalog?q=` | 카탈로그 목록/검색 | — |
| POST | `/dataops/catalog` | 신규 아카이브(사용자 소스) 등록 | — |
| GET | `/dataops/catalog/{source_id}` | 단일 소스 메타데이터 | — |
| DELETE | `/dataops/catalog/{source_id}` | 사용자 소스 삭제(시드 보호) | — |
| GET | `/dataops/{source_id}` | 조회(filter·sort·page·page_size) | `data:read` |
| POST/PUT/PATCH/DELETE | `/dataops/{source_id}` | CRUD | `data:write` |

### MLOps 모니터링
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/monitoring/metrics` | 6대 지표 시계열 + latest |
| POST | `/monitoring/metrics/regression` | MSE·MAE 계산 |
| POST | `/monitoring/metrics/classification` | Accuracy·Precision·Recall·F1 계산 |
| GET | `/monitoring/drift?drifted=&model_id=&auto_retrain=` | 시드 분포 PSI/KL 판정(+선택적 재학습 발화) |
| POST | `/monitoring/drift` | 임의 분포 PSI/KL 판정 |
| POST | `/monitoring/outliers?method=zscore\|iqr` | 이상치 탐지 |
| GET | `/monitoring/explain` | SHAP 특징 중요도(시드) |
| POST | `/monitoring/explain` | 기여도 → 중요도 정렬 |

### MLOps 오케스트레이션
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/orchestration/models` | 등록 모델·현재 버전·지표 |
| GET | `/orchestration/runs` | 실행 이력(최신 우선) |
| POST | `/orchestration/events` | 재학습 이벤트 → 상태머신 실행 |

---

## 6. 인증

- **HS256 JWT**(stdlib `hmac`/`hashlib`, pyjwt 미사용). payload: `{sub, scope:"data:read data:write", source, iat, exp(1h)}`.
- **OAuth2**: Authorization Code Grant 흐름(access_token은 JWT 형식).
- `/dataops/{source_id}` 접근은 `Authorization: Bearer <token>` 필수, 미인증 시 401.
- **토큰 발급 prod 게이트**(`require_client`): `XOPS_ENVIRONMENT=prod`이고 `client_id`가 설정된 경우에만 `X-Client-Id`/`X-Client-Secret` 검증. dev는 개방(데모 편의).

---

## 7. 영속화 (SQLite)

`src/core/db.py` — 단일 파일 SQLite(WAL). 재시작·멀티워커에서 상태 일관성 확보.

| 테이블 | 내용 |
|---|---|
| `user_sources` | 사용자 등록 아카이브 스키마(JSON) |
| `model_versions` | 모델 버전 오버라이드(승급 반영) |
| `runs` | 재학습 실행 이력(JSON) |

- 시드 카탈로그·시계열은 `mock_data.json`(읽기전용)이 담당. **EventBus debounce는 프로세스 국소**(단기 anti-spam, 재시작 시 리셋 허용).
- DB 파일 기본 경로 `data/xops.db`(`.gitignore` 대상), `XOPS_DB_PATH`로 변경. 테스트는 임시 DB로 격리.
- 검증: 크로스 프로세스 재시작 후 사용자 소스·모델 버전(v3.1)·실행 이력 보존 실측.

---

## 8. 설정 (`XOPS_` 환경변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `XOPS_ENVIRONMENT` | `dev` | `prod`면 기본 JWT 시크릿 사용 시 기동 거부 + 토큰 게이트 활성 |
| `XOPS_JWT_SECRET` | (dev 기본값) | **prod 필수 교체** |
| `XOPS_CLIENT_ID` / `XOPS_CLIENT_SECRET` | 빈값 | prod 토큰 발급 자격증명 |
| `XOPS_CORS_ORIGINS` | `localhost:5173` | 허용 오리진 |
| `XOPS_DB_PATH` | `data/xops.db` | SQLite 경로 |
| `XOPS_PSI_THRESHOLD` / `XOPS_KL_THRESHOLD` | 0.2 / 0.1 | 드리프트 임계 |
| `XOPS_ZSCORE_THRESHOLD` / `XOPS_IQR_MULTIPLIER` | 3.0 / 1.5 | 이상치 임계 |
| `XOPS_ROLLBACK_LATENCY_MS` | 200 | 자동 롤백 임계 지연 |
| `XOPS_RETRAIN_MIN_INTERVAL_MINUTES` | 30 | 재학습 debounce 간격 |

`.env.example` 참고, `.env`로 오버라이드.

---

## 9. 프론트엔드 연동

`platform/frontend`(React+Vite)의 DataOps·Monitor·Orchestrator 3개 페이지가 실제 `/api/v3` 호출로 동작(mock 제거).

- `src/lib/api.js` — native fetch 래퍼(base=`VITE_API_BASE_URL`, 비-2xx throw, Bearer 주입)
- dev: `vite.config.js`의 proxy `/api → localhost:8000`(CORS 회피). prod: nginx가 동일 역할.
- **DataOps**: 카탈로그·토큰 발급·CRUD·아카이브 등록/삭제를 서버 호출. `buildQuery`(SQL/MQL 미리보기)만 클라이언트 잔존.
- **Monitor**: 지표·드리프트(PSI 실계산값)·SHAP를 서버 조회.
- **Orchestrator**: [실행] → `POST /orchestration/events`, 반환 `PipelineRun`으로 승급/롤백 반영(스텝퍼는 UX 애니메이션). 드리프트 재학습은 `injectDrift` 단일 경로.

---

## 10. 테스트

- **unit**: query_builder(쓰기 바인딩 SQL 포함), safety(주입·쓰기 값 검증), adapters(fake 커넥션 실행 경로),
  jwt, client_gate, compose 계약, metrics/drift/outliers/evaluator/deployer/events, training, persistence
- **integration**: dataops·monitoring·orchestration API, catalog CRUD, 카탈로그 주입 차단,
  실 저장소 어댑터 주입(읽기·쓰기·degrade), enhancements(드리프트→재학습·filter 검증·prod 시크릿)
- **총 183건 통과 / 커버리지 96.4%** (게이트 80%)

---

## 11. 배포 (Docker)

- `backend/xops-service/Dockerfile` — python:3.11-slim, requirements(+DB 드라이버) 설치, uvicorn 8000, `/app/data` 볼륨
- `frontend/Dockerfile` — 멀티스테이지(node build → nginx), `nginx.conf`가 `/api`를 `xops-service:8000`으로 프록시 + SPA 폴백
- `platform/compose.xops.yaml` — 프론트(8080) + 백엔드(8000) + 실 저장소 3종(postgis 5433·timescale 5434·mongo 27017) + named volume 4개.
  호스트 포트 충돌 시 `XOPS_HOST_PORT`/`XOPS_FRONTEND_HOST_PORT` 로 오버라이드(내부 포트·프록시 계약 불변)
- prod 전환 시 `XOPS_ENVIRONMENT=prod` + `XOPS_JWT_SECRET`(+선택 `XOPS_CLIENT_ID/SECRET`) 주입

> 현 상태: **`docker compose up -d --build` 완전체 실물 기동 검증 완료(2026-09-01)** — 컨테이너 경유로
> 헬스·카탈로그 9종·실 저장소 3종 조회(`source_kind=database`)·쓰기 왕복(SQL 17→18→17행, MQL POST/PATCH/DELETE)을 실측했다.

---

## 12. 스코프 밖 / 후속

- Overview·Simulator·Reporter 페이지 연동
- report-service(리포팅/내보내기), dashboard-service(조회/집계)
- 로그인 기반 인증, 오케스트레이션 비동기 run + 상태 폴링, E2E(Playwright) 자동화

### 해소된 항목 (2026-08-31 ~ 09-01)

- ~~실제 Postgres/PostGIS/Mongo/Timescale 연결~~ → **완료.** `QueryAdapter` seam 을 실제 실행
  계층(backends.py)으로 채우고 compose 에 실 저장소 3종을 편성했다. 읽기·쓰기(파라미터 바인딩)
  모두 실 저장소를 관통하며, DSN·드라이버 부재 시 In-Memory 로 degrade 하는 이중 구조는 유지된다.
  실데이터 2건(행정구역경계 251·사회복지시설 17)도 적재·카탈로그 등록됐다(ds_08·ds_09,
  `platform/data/real/README.md`).
- ~~compose 실물 기동 검증~~ → **완료.** §11 참조.
- MLflow/MinIO 는 재학습 파이프라인 실물화(SQLite 레지스트리 + 로컬 아티팩트) 때 **의도적으로
  도입하지 않았다** — 필요 시 `registry.py` provider seam 에 어댑터로 끼운다(체크리스트 참조).
