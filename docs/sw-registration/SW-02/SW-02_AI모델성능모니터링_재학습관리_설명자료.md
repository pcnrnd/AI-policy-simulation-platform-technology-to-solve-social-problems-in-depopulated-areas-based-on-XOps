> 초안 (2026-09-23 개정, 미확정) — 제출·신청·게재 확정본이 아님

# 인공지능 모델 성능 모니터링 및 재학습 관리 프로그램 — 설명자료

- 문서번호: SW-02
- 작성일: 2026-09-11 / 개정: 2026-09-23 (기준 커밋 갱신, 실데이터 브리지 반영, 단독 진입점 추가)
- 작성자: PCN R&D
- 대상 코드베이스: `F:/rnd_github/pcnrnd/Population_decline`, 기준 브랜치 `master`, 기준 커밋 `2e71da4224cc4a6970b6888d5e6968eb243dd7ae` (이전 초안 `f81df7ad8d82b37da9af71e31fa962ba58fa4c9a`)
- 관련 기술문서: TD-02(인공지능 모델 모니터링 및 이벤트 기반 재학습 기술서)
- 기능 경계: [../README.md](../README.md#기능-경계)

## 1. 목적

예측 모델의 성능(6대 지표)·분포 이동(드리프트)·이상치·특징 기여도를 계산하고, 성능저하·드리프트 신호를 받아 재학습→평가→승급 판정→버전 기록까지 프로세스 안에서 자동으로 잇는 프로그램이다. 외부 서빙 배포나 실제 트래픽 전환은 수행하지 않으며, 승급·유지 결과를 내부 모델 레지스트리에 기록한다.

## 2. 주요기능

| 기능 | 설명 | 근거 |
|---|---|---|
| 6대 지표 계산·조회 | 회귀(MSE·MAE)·분류(Accuracy·Precision·Recall·F1)를 순수 Python으로 계산. 조회는 `window`로 마지막 N개 관측만 반환 | `src/mlops/monitoring/metrics.py`, `GET /monitoring/metrics` |
| 실측 지표 출처 분리 | 학습 실행 기록(`runs`)·아티팩트 JSON에서만 실측 지표·계수를 읽고, 없으면 빈 값으로 둔다(시드로 메우지 않음) | `src/mlops/monitoring/sources.py` |
| 드리프트 감지(PSI/KL) | 참조·현재 분포로 Population Stability Index·KL-Divergence 계산, 임계 초과 판정 | `src/mlops/monitoring/drift.py` |
| 이상치 탐지 | Z-score·IQR 두 방식 | `src/mlops/monitoring/outliers.py` |
| 설명가능성(특징 중요도) | SHAP 설치 시 SHAP, 미설치 시 평균 절대 기여도 fallback | `src/mlops/monitoring/explain.py` |
| 실데이터 모델 조회 브리지 | `data:read` 토큰이 있고 `include_seed=false`인 호출에 한해 실데이터 모델의 평가·드리프트·학습 job을 기존 모니터링·오케스트레이션 응답 스키마로 제공(`source="realdata"`) | `src/mlops/monitoring/realdata_bridge.py` |
| 이벤트 기반 재학습 트리거 | 드리프트/성능저하/수동 3종 트리거, model_id별 debounce(수동은 항상 통과) | `src/mlops/orchestration/events.py` |
| 비동기 재학습 실행 | 이벤트를 202로 접수하고 백그라운드에서 실행, 진행 상태를 저장해 프로세스 재기동 시 미종결 실행을 마감 | `POST /orchestration/events`, `registry.recover_stale_runs` |
| 재학습 상태머신 | queued→preparing→training→evaluating→deploying→succeeded/rolled_back/rejected/debounced | `src/mlops/orchestration/orchestrator.py` |
| 승급 판정 | 두 모델에 공통인 최우선 지표(f1>accuracy>mae>mse) 하나로 비교 | `src/mlops/orchestration/evaluator.py` |
| 배포 판정 | 후보 예측 지연이 임계(`rollback_latency_ms`, 기본 200ms)를 넘으면 직전 버전 유지 | `src/mlops/orchestration/deployer.py` |
| ML 파이프라인 등록·실행 | 파이프라인 등록·삭제·실행, 실행 이력·로그 조회 | `/orchestration/pipelines*`, `/orchestration/runs*` |
| 그리드 학습(LOO 교차검증) | lag×ridge_lambda 그리드를 학습해 최고 후보 선택, 아티팩트 저장 | `src/mlops/training/trainer.py`, `model.py` |
| 모델 레지스트리 | 운영 버전·아티팩트·실행 이력의 단일 출처(SQLite 영속화) | `src/mlops/orchestration/registry.py` |

드리프트 초과 시 `auto_retrain=true` 옵션으로 재학습을 즉시 발화하는 경로(`monitoring.py`의 `_maybe_retrain`)가 모니터링과 재학습을 연결한다.

## 3. 특징

- **외부 수치 라이브러리 없이 순수 Python 구현** — 6대 지표·PSI/KL·Z-score/IQR·릿지 회귀(정규방정식 가우스 소거)가 표준 라이브러리만으로 동작한다.
- **지표의 출처 등급 구분** — 승급 판정의 현행(기준선) 값은 실측(`trained`) 지표만 인정하고, 명시 주입값·파생 fallback 값은 다음 판정의 기준으로 쓰지 않는다(`ModelRegistry._measured`). 응답의 `source`로 시드(`seed`)·시드 학습 실측(`measured`)·실데이터(`realdata`)를 구분한다.
- **단일 primary 지표 승급 규칙** — 6개 지표를 동시에 비교하지 않고 공통 최우선 지표 하나로 승급을 결정해 판정 모호성을 없앤다.
- **학습 실패에 대한 안전망** — 학습이 실패해도 결정적 파생값으로 상태머신이 멈추지 않으며, 어느 경로(`explicit`/`trained`/`derived`)로 승급했는지 실행 기록에 남는다.

## 4. 사용방법

단독 실행은 [README.md](README.md)를 따른다. 요약:

```powershell
Set-Location platform/backend/xops-service
python -m pip install -r requirements.txt
python -m uvicorn sw_main:create_sw02_app --factory --port 8002
```

```powershell
curl "http://localhost:8002/api/v3/monitoring/metrics?include_seed=true&window=12"
curl "http://localhost:8002/api/v3/monitoring/drift?include_seed=true&drifted=true&model_id=population-forecast&auto_retrain=true"
curl -X POST "http://localhost:8002/api/v3/orchestration/events" -H "Content-Type: application/json" -d "{\"model_id\": \"population-forecast\", \"trigger\": \"manual\"}"
curl "http://localhost:8002/api/v3/orchestration/runs/<run_id>"
curl "http://localhost:8002/api/v3/orchestration/models?include_seed=true"
```

`include_seed` 기본값은 false이며, 이때 실측이 없으면 빈 응답이다.

## 5. 실행환경

| 항목 | 내용 |
|---|---|
| 언어·런타임 | Python 3.10 이상 |
| 필수 의존성 | fastapi>=0.115, uvicorn[standard]>=0.30, pydantic>=2.6, pydantic-settings>=2.2 |
| 선택 의존성(ml) | numpy>=1.26, shap>=0.44 (미설치 시 설명가능성만 fallback, 지표·드리프트·학습은 항상 동작) |
| 선택 의존성(db) | psycopg[binary]>=3.1 — 실데이터 브리지가 PostgreSQL `rd_*`를 읽을 때만 필요. 없으면 실데이터 항목이 비어 있고 나머지는 동작 |
| 저장소 | SQLite(`model_versions`·`model_artifacts`·실행 이력, `XOPS_DB_PATH`), 학습 아티팩트 JSON(`XOPS_MODEL_ARTIFACT_DIR`) |
| 실행 방식 | 단독: `sw_main:create_sw02_app` / 통합 플랫폼: `main:app`, Docker Compose(`platform/compose.xops.yaml`) |

## 6. 코드 범위

| 구분 | 경로 (`platform/backend/xops-service/` 기준) |
|---|---|
| 단독 진입점 | `sw_main.py`(`create_sw02_app`) |
| 모니터링 모듈 | `src/mlops/monitoring/{metrics,drift,outliers,explain,sources,realdata_bridge}.py` |
| 오케스트레이션 모듈 | `src/mlops/orchestration/{events,orchestrator,evaluator,deployer,registry}.py` |
| 학습 모듈 | `src/mlops/training/{dataset,model,trainer}.py` |
| API 라우터 | `src/api/v3/{monitoring,orchestration}.py` |
| 요청/응답 스키마 | `src/schemas/{monitoring,orchestration}.py` |
위 표가 등록 범위다(모두 `src/mlops/{monitoring,orchestration,training}/*`·라우터·스키마). 아래는 **등록 범위 밖 공용 의존**으로, 기동에 필요하지만 SW-02 등록 범위에 넣지 않는다(루트 결정 2026-09-23).

| 공용 의존 | 경로 |
|---|---|
| 공통 기반(SW-01과 공유) | `src/core/*`, `src/auth/jwt.py`, `src/api/dependencies.py` |
| 실데이터 모듈(`realdata_bridge` 경유) | `src/realdata/{candidates,jobs,monitoring,snapshot,pg_reader,errors,models,evaluate,features}.py`, `namwon_dong_map.json` |
| SW-01 모듈 1개 | `src/dataops/safety.py`(`pg_reader`가 쓰는 식별자 검증 함수 `assert_safe_identifier`) |

`src/realdata/`의 기능과 `/api/v3/realdata/*` 라우터, 분석·규칙(`analysis.py`·`rules.py`)은 SW-02 기능으로 서술하지 않는다. SW-02 기능 서술은 브리지를 거친 조회까지다.

`f81df7a` 대비 범위 내 변경: `realdata_bridge.py`·`sources.py` 신규, `orchestrator.py`·`registry.py`·`training/model.py` 수정, `api/v3/monitoring.py`·`orchestration.py`(실데이터 브리지·`window`·202 접수), `schemas/orchestration.py`.

## 7. 2025년 등록 「데이터 워크플로우 관리 프로그램」과의 차별점

2025-11-07 등록 SW는 Airflow DAG·DVC 버전관리·컨테이너 재현성 등 **DataOps 실행 기반**에 해당하며(위키 §7), 모델 성능 모니터링·재학습 판정 로직을 다루지 않는다. SW-02는 학습된 모델의 지표·드리프트를 감시하고 승급/유지를 결정하는 **MLOps 판단 로직**으로, 2025 등록 범위와 코드·문제의식이 겹치지 않는다.

| 구분 | 2025 등록 SW | SW-02(본 신규) |
|---|---|---|
| 핵심 자산 | 파이프라인 실행 스케줄링·데이터 버전관리·재현성 | 모델 지표 계산·드리프트 판정·승급/유지 상태머신 |
| 대상 데이터 | 파이프라인이 다루는 원천 데이터 전반 | 모델 예측 결과(y_true/y_pred)와 학습 시계열 |
| 코드 위치 | Airflow DAG·오퍼레이터(본 저장소 범위 밖 — 위키 참조) | `src/mlops/*`(본 저장소, 2026-06 이후 커밋) |

[확인 필요] 2025 등록 명세서 원문을 직접 열람하지 않았으며, 비교는 `population-decline-history.md` §7 요약 인용에 근거한다.

## 8. 독립 실행 가능성

- `sw_main:create_sw02_app`은 `/api/v3/monitoring`·`/api/v3/orchestration`만 등록한 FastAPI 앱을 띄운다. SW-01의 라우터·카탈로그·CRUD 모듈은 import하지 않는다. 유일한 예외는 `realdata/pg_reader.py`가 쓰는 `src/dataops/safety.py`이며, 테스트가 이 한 모듈만 허용하고 나머지 `src.dataops.*` 로드를 실패로 본다(`tests/integration/test_sw_entrypoints.py`).
- 실데이터 조회에 필요한 `data:read` 토큰은 SW-02가 발급하지 않는다. 같은 `XOPS_JWT_SECRET`으로 SW-01(또는 통합 플랫폼)이 발급한 토큰을 쓴다. 토큰 없이도 시드·실측 지표 조회와 재학습은 동작한다.
- 통합 플랫폼 실행(`main:app`, Compose)은 그대로 전체 라우터를 한 프로세스로 띄운다.
- **[확인 필요]** SW 등록 신청서의 "독립적으로 실행 가능한 프로그램" 기준 충족 여부는 신청서 서식과 담당자 확인이 필요하다.
