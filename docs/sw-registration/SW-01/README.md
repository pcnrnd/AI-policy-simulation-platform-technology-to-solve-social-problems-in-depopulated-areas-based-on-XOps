# SW-01 단독 실행

메타데이터 기반 다기종 데이터 아카이빙 관리 프로그램을 `/api/v3/dataops` 라우터만으로 띄운다. 기능 경계는 [../README.md](../README.md#기능-경계).

## 실행

저장소 루트에서 시작한다. Python 3.10 이상.

```powershell
Set-Location platform/backend/xops-service
python -m pip install -r requirements.txt
python -m uvicorn sw_main:create_sw01_app --factory --port 8001
```

- 실행 디렉터리는 `xops-service/`여야 한다(`from src...` 임포트와 `.env` 해석).
- 상태 확인: `http://localhost:8001/` → `{"xops":"connected","program":"sw01-dataops"}`, API 문서 `/docs`.
- 개발 DB를 건드리지 않으려면 기동 전에 `$env:XOPS_DB_PATH`를 임시 경로로 지정한다.
- 실 저장소 연결(선택): `python -m pip install -r requirements-db.txt` 후 `XOPS_PG_DSN`·`XOPS_TIMESCALE_DSN`·`XOPS_MONGO_URI`. 미지정이면 In-Memory 응답(`source_kind="in-memory"`)이며 실제 저장소 조회 증거가 아니다.

## 확인 순서

```powershell
curl "http://localhost:8001/api/v3/dataops/catalog"
curl -X POST "http://localhost:8001/api/v3/dataops/token/<source_id>"
curl -H "Authorization: Bearer <TOKEN>" "http://localhost:8001/api/v3/dataops/<source_id>?page=1&page_size=20"
```

`<source_id>`는 카탈로그 응답의 `id`를 쓴다. 카탈로그 쓰기·데이터 CRUD는 JWT scope를 검사한다. dev에서는 토큰 발급이 열려 있고 prod(`XOPS_ENVIRONMENT=prod`)는 `XOPS_CLIENT_ID`·`XOPS_CLIENT_SECRET`을 검사한다.

## 검증

```powershell
python -B -m pytest -p no:cacheprovider -o addopts="" tests/integration/test_sw_entrypoints.py
```

라우터 경계(`dataops`만 등록)와 import 독립성(`src.mlops`·`src.realdata` 미로드)을 확인한다. 전체 테스트는 저장소 `CLAUDE.md`의 검증 절차를 따른다.

## 통합 플랫폼과의 관계

`main:app`(Compose 포함)은 SW-01·SW-02와 Overview·realdata 라우터를 함께 띄운다. 이 단독 진입점은 그 동작을 바꾸지 않는다. 적재 스크립트(`platform/scripts/load_real_data.py`, `load_external_data.py`)는 DB를 변경하므로 상태 점검과 구분해 실행한다.
