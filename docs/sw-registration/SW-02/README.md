# SW-02 단독 실행

인공지능 모델 성능 모니터링 및 재학습 관리 프로그램을 `/api/v3/monitoring`·`/api/v3/orchestration` 라우터만으로 띄운다. 기능 경계는 [../README.md](../README.md#기능-경계).

## 실행

저장소 루트에서 시작한다. Python 3.10 이상.

```powershell
Set-Location platform/backend/xops-service
python -m pip install -r requirements.txt
python -m uvicorn sw_main:create_sw02_app --factory --port 8002
```

- 실행 디렉터리는 `xops-service/`여야 한다.
- 상태 확인: `http://localhost:8002/` → `{"xops":"connected","program":"sw02-mlops"}`, API 문서 `/docs`.
- 재학습은 SQLite(`XOPS_DB_PATH`)와 아티팩트(`XOPS_MODEL_ARTIFACT_DIR`)에 기록한다. 개발 중인 모델·실행 이력을 보존하려면 기동 전에 두 변수를 임시 경로로 지정한다.
- 선택: `pyproject.toml`의 `ml` 선택 의존성(numpy·shap) 설치 시 SHAP 사용. 실데이터 모델 조회는 `requirements-db.txt`와 `XOPS_PG_DSN`, `data:read` 토큰이 필요하다(토큰은 같은 `XOPS_JWT_SECRET`의 SW-01/통합 플랫폼이 발급).

## 확인 순서

```powershell
curl "http://localhost:8002/api/v3/monitoring/metrics?include_seed=true&window=12"
curl "http://localhost:8002/api/v3/monitoring/drift?include_seed=true&drifted=true&model_id=population-forecast"
curl -X POST "http://localhost:8002/api/v3/orchestration/events" -H "Content-Type: application/json" -d "{\"model_id\": \"population-forecast\", \"trigger\": \"manual\"}"
curl "http://localhost:8002/api/v3/orchestration/runs/<run_id>"
curl "http://localhost:8002/api/v3/orchestration/models?include_seed=true"
```

- `POST /orchestration/events`는 202와 `run_id`를 돌려주고 백그라운드에서 실행한다. 종결 상태는 `GET /orchestration/runs/<run_id>`로 확인한다. 이 호출은 저장 상태를 바꾼다.
- `include_seed` 기본값은 false이며 실측이 없으면 빈 응답이다. `include_seed=true`의 시드 응답은 운영 모델의 실시간 수집 결과가 아니다.
- 승급·유지는 프로세스 내부 레지스트리 기록이며 외부 서빙 배포·트래픽 전환을 하지 않는다.

## 검증

```powershell
python -B -m pytest -p no:cacheprovider -o addopts="" tests/integration/test_sw_entrypoints.py
```

라우터 경계(`monitoring`·`orchestration`만 등록)와 import 독립성(`src.dataops`는 `safety`만 허용)을 확인한다. 전체 테스트는 저장소 `CLAUDE.md`의 검증 절차를 따른다.

## 통합 플랫폼과의 관계

`main:app`(Compose 포함)은 SW-01·SW-02와 Overview·realdata 라우터를 함께 띄운다. 이 단독 진입점은 그 동작을 바꾸지 않는다.
