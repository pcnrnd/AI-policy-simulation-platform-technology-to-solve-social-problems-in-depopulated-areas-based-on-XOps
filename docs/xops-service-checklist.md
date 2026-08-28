# xops-service 개발내용 체크리스트

> 원 개발내용(명세) 대비 구현 완료 현황 · 최종 갱신 2026-08-28
> 범례: ✅ 완료(xops-service) · 🟡 완료(단, 실 인프라는 seam/후속) · 🔷 프론트엔드 담당(백엔드 범위 밖) · ⬜ 미착수

---

## 1. DataOps — 데이터 라이프사이클 관리

### 가) 메타데이터를 이용한 다기종 데이터 관리 (빅데이터 관리 아카이빙)
- [x] ✅ 데이터 가상화 고려 다기종 데이터 관리 — `dataops/catalog.py`, `adapters.py` (PostgreSQL/PostGIS/Mongo/Timescale)
- [x] ✅ 메타데이터 기반 수집·가공 데이터 관리, API로 저장소 접근 제어 — `MetadataCatalog` + `/dataops/*` (직접 저장소 접근 차단)
- [x] ✅ API 요청 → 메타데이터 검색 → 저장소 정보 획득 → 적합 DB Adapter + SQL 호출 — `service.DataService`
- [x] ✅ SQL 실행 → 결과 API 응답 — `buildApiResponse` 계약(표준 REST JSON)
- [x] ✅ API 통한 저장소 관리로 보안 강화 (사용자는 저장소 비인지) — JWT 게이트 + Adapter 추상화
- [x] ✅ 아카이빙(티어 Hot/Warm/Cold·보존정책·적재일) — `archive_meta` 응답 + 카탈로그
- [x] ✅ 메타데이터 적재범위(range) 자동 주입 — SQL `BETWEEN` / MQL `$gte·$lte`

### 나) 데이터 처리 API 기술 (Data API 빌더)
- [x] ✅ 표준 SQL 설정 기반 데이터 처리(CRUD) — `query_builder.py`
- [x] 🟡 In-memory 기반 처리로 속도 보장 — `InMemoryAdapter`(결정적) + 응답속도 배지. **실제 DB 연결은 `QueryAdapter` Protocol seam(후속)**
- [x] ✅ API생성기 → 표준 REST 응답 — 5개 메서드 라우트 + 발급 API 목록(프론트)
- [x] ✅ CRUD(POST/GET/PUT/PATCH/DELETE) — `/dataops/{source_id}` 전 메서드
- [x] ✅ 필터링·정렬·페이징 REST — `filter`·`sort`·`page`·`page_size` 쿼리
- [x] ✅ JWT / OAuth2 인증 — `auth/jwt.py` (`/token`·`/oauth2`)
- [x] ✅ 데이터 요청 관리·기록 — 구조화 JSON 로깅 + 발급 API 목록 보존
- [x] ✅ 다기종 확장(NoSQL/MQL) — Mongo 소스 `db.col.find({..$gte,$lte})` 재현
- [x] ✅ 사용자 아카이브 등록·삭제 (메타데이터 등록 단계) — `POST/DELETE /dataops/catalog`, SQLite 영속화

---

## 2. MLOps — 모델 배포 자동화

### 가) 인공지능 모델 성능 모니터링
- [x] ✅ 통합 모니터링(사일로 제거·가시성) — `/monitoring/*` + 프론트 대시보드
- [x] ✅ 모델 Score 6대 지표 (MSE·MAE·F1·Accuracy·Recall·Precision) — `metrics.py` 실계산
- [x] ✅ 드리프트 모니터링(데이터 분포 변화) — `drift.py` PSI(≥0.2)·KL(≥0.1) 실계산
- [x] ✅ 이상값(Outlier) 모니터링 — `outliers.py` Z-score·IQR
- [x] ✅ 설명 가능 데이터 모니터링(feature 중요도·XAI) — `explain.py` SHAP + fallback
- [x] ✅ 통계 데이터 분석 → 최적화·지속 개선 — 지표/드리프트/이상치 실계산 + 드리프트 감지→실제 재학습→실측 지표 승급 판정으로 개선 루프가 닫힘(`training/` + `orchestrator.py`)

### 나) 오케스트레이션 기술
- [x] ✅ 이벤트 기반 오케스트레이션 — `events.py`(EventBus·debounce) + `/orchestration/events`
- [x] ✅ 성능저하(드리프트 등) 감지 → 재학습 → 배포 자동화 — `orchestrator.py` 상태머신
- [x] ✅ 승급 판정(최고 성능 자동 선택 = SOTA) — `evaluator.py` (f1>acc>mae>mse)
- [x] ✅ canary→full 배포 + 자동 롤백(latency>200ms) — `deployer.py`
- [x] ✅ 드리프트 감지 → 재학습 자동 발화 — `/monitoring/drift?auto_retrain` + 프론트 injectDrift 경로
- [x] ✅ SOTA 모델 도출 — `training/`에서 하이퍼파라미터 그리드(lag×λ 6후보)를 **실제 학습**하고 LOO 교차검증 예측으로 6지표를 실측해 최고 후보를 선택. 승급 기준은 절편-only 기준선(최초) → 직전 실측 아티팩트(이후)로 올라간다. 모델 레지스트리는 SQLite(`model_artifacts`) + 로컬 JSON 아티팩트 — **MLflow/MinIO 없이 완결**

---

## 3. 사회문제 해결지원 시뮬레이션 (참고 — xops-service 범위 밖)

> UI/UX·공간정보는 프론트엔드(`platform/frontend`) 담당. xops-service(백엔드 DataOps+MLOps)의 범위가 아니므로 별도 표기.

- [ ] 🔷 사용자 관점 UI/UX 설계(전자정부 가이드라인) — 프론트 `SimulatorPage`·`styles/accessibility.css`
- [ ] 🔷 공간정보 지도 표시(지도 API) — 프론트 Leaflet + VWorld
- [ ] 🔷 위치 데이터·상세 표출, 레이어 적층 표시 — 프론트 지도 레이어 토글

---

## 요약

| 부문 | 완료 | 비고 |
|---|---|---|
| DataOps | ✅ 전 항목 | In-memory 처리(실 DB는 seam) |
| MLOps 모니터링 | ✅ 전 항목 | 6지표·드리프트·이상치·XAI 실계산 |
| MLOps 오케스트레이션 | ✅ 전 항목 | 인프로세스 실 학습·실측 승급(외부 학습 인프라 불요) |
| 시뮬레이션 UI/공간정보 | 🔷 프론트 담당 | xops 범위 밖 |

**xops-service 담당 범위(DataOps + MLOps)의 명세 항목은 전부 구현 완료.** 테스트 117건 / 커버리지 96.9%.

### 🟡·후속으로 남긴 실 인프라 연동
- 실제 Postgres/PostGIS/Mongo/Timescale 연결 (현재 In-Memory + Adapter seam)
- 배포: `docker compose up` 실물 기동 검증(현재 파일 정적 검증만)

### 실물화로 해소된 항목 (재학습 파이프라인)
「실제 모델 재학습 파이프라인」은 `src/mlops/training/`(순수 Python 릿지 회귀 + LOO 실측)과
SQLite 모델 레지스트리로 구현돼 후속 목록에서 제외했다. MLflow(실험 추적)·MinIO(아티팩트)는
현재 UI가 요구하는 기능(재학습·버전관리·승급·롤백)을 이 구성이 이미 충족하므로 **의도적으로 도입하지
않았다** — 필요해지면 `registry.py`의 provider seam에 어댑터로 끼운다.

상세 설계는 [xops-service.md](xops-service.md) 참조.
