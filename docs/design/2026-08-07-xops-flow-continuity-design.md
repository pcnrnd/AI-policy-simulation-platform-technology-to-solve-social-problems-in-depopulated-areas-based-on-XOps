# XOps 탭 간 플로우 연속성 개선 설계

- 날짜: 2026-08-07
- 대상: `platform/frontend` (React SPA)
- 목적: 시연/심사 데모에서 **DataOps → MLOps 성능 모니터 → 오케스트레이터 → 정책 시뮬레이터 & 추천** 4개 탭이 하나의 XOps 스토리로 이어져 보이게 한다.

## 배경 (현황 분석)

- 유일하게 완성된 크로스탭 흐름: 모니터 드리프트 주입 → 컨텍스트 `startPipeline` → 오케스트레이터 파이프라인 → 승급 시 모니터/대시보드 지표 갱신 (백엔드 실연동 포함). **이 체인은 유지하고 건드리지 않는다.**
- 단절 지점:
  1. 모니터의 모델 셀렉트는 표시용 — 드리프트는 항상 `population-forecast` 하드코딩, 재학습도 `RETRAIN_PIPELINES[0]` 고정
  2. DataOps는 완전 고립 — 카탈로그·발급 API가 다른 탭으로 흐르지 않음
  3. 시뮬레이터는 어떤 모델·데이터 기반인지 표시 없음 — 승급된 모델 버전이 전달되지 않음

## 범위·제약

- **4개 탭만**: DataOps, 모니터, 오케스트레이터, 시뮬레이터 (리포터는 담당 분리로 제외 — 시뮬레이터→리포터 연결은 리포터 담당자 몫, 종합 대시보드는 현상 유지)
- **프론트만 변경**, 백엔드(xops-service) 무변경
- **모킹데이터 전제**: 연결은 전역 Context(메모리) + 기존 localStorage 패턴으로 구현. DB/Redis 불필요 — 탭들은 단일 SPA 컴포넌트라 상태 공유는 `AppStateContext`로 충분. 다중 사용자 실시간 공유가 필요해지는 시점(WebSocket/SSE + pub/sub)에 재검토.
- 기존 페이지 레이아웃 재배치 없음 — 추가만 한다.

## 설계

### 1. 전역 플로우 리본 — `components/XopsFlowRibbon.jsx` (신규)

```
[① DataOps]──[② 성능 모니터]──[③ 오케스트레이터]──[④ 정책 시뮬레이터]
 카탈로그 5건   ⚠ 드리프트 감지    ⟳ 재학습 중(3/6)     모델 v1.3 적용
```

- `App.jsx`에서 activeTab이 4개 탭 중 하나일 때만 content-body 상단에 마운트
- 현재 탭 하이라이트, 단계 클릭 시 `setActiveTab` 이동
- 라이브 배지는 기존 전역 상태 재사용: `driftInjected`, `pipelineRunning`/`pipelineStep`, `modelStore`
- DataOps 건수: 리본 마운트 시 1회 `GET /api/v3/dataops/catalog` — 실패 시 배지 생략(콘솔로그만)

### 2. 다음 단계 CTA 배너 — `components/NextStepBanner.jsx` (신규, 공용 1개)

이벤트 발생 시에만 등장:

| 위치 | 조건 | 문구 → 이동 |
|---|---|---|
| 모니터 | 드리프트 감지/재학습 중 | "자동 재학습 진행 중 — 오케스트레이터에서 보기 →" |
| 오케스트레이터 | 승급 완료 | "vX.Y 승급 완료 — 정책 시뮬레이터에서 활용 →" |
| DataOps | API 발급 성공 | "이 데이터로 학습된 모델 모니터링 →" |

### 3. 모델 컨텍스트 전역화

- `AppStateContext`에 `selectedModelId` 추가 (기본 `population-forecast`)
- 모니터 모델 셀렉트가 전역 값 읽기/쓰기 → `injectDrift`가 선택 모델로 이벤트 발화
- `startPipeline`이 선택 모델에 매칭되는 파이프라인 실행 — `constants/models.js`에 모델↔파이프라인 매핑 키 추가
- 오케스트레이터에서 선택 모델의 파이프라인·Model Store 행 하이라이트

### 4. 데이터 계보(lineage)

`MODEL_REGISTRY` 각 모델에 `dataSources: [카탈로그 source_id]` 추가 — 이 한 곳의 매핑으로:

- **모니터**: 드리프트 카드에 "원인 데이터: {소스명}" 칩 → 클릭 시 DataOps 이동 + 해당 소스 자동 선택 (1회성 전역 값 `pendingCatalogFocus`)
- **DataOps**: 카탈로그 행에 "사용 모델" 칩(역매핑) → 클릭 시 모니터 이동 + `selectedModelId` 설정
- **시뮬레이터**: STAGE① "연계 소스 n개"를 실제 카탈로그 소스 이름으로 표시 + DataOps 링크

### 5. 모델 버전 전파

- 시뮬레이터 STAGE① 실행 바에 "예측 모델: {name} {version}" 칩 — `modelStore`의 해당 모델 최신 운영 버전
- 이번 세션에서 승급된 버전이면 "✦ 오늘 승급" 강조

## 에러 처리

- 백엔드 다운: 기존 mock 폴백 유지. 리본 배지·lineage 칩은 조용히 생략, `addConsoleLog`로만 기록
- lineage 매핑이 없는 모델/소스: 칩 미표시 (에러 아님)

## 변경 파일

| 구분 | 파일 |
|---|---|
| 신규 | `components/XopsFlowRibbon.jsx`, `components/NextStepBanner.jsx`, 관련 CSS |
| 수정 | `context/AppStateContext.jsx`, `constants/models.js`, `App.jsx`, `pages/MonitorPage.jsx`, `pages/OrchestratorPage.jsx`, `pages/DataOpsPage.jsx`, `pages/SimulatorPage.jsx` |

## 검증 (테스트 러너 없음 → 빌드 + 데모 시나리오)

1. `npm run build` 통과
2. 데모 시나리오 체크리스트:
   - [ ] DataOps: 카탈로그 소스에 "사용 모델" 칩 표시, 클릭 시 모니터 이동+모델 선택
   - [ ] 모니터: 모델 셀렉트 변경 → 드리프트 주입 → 리본에 ⚠ 표시, CTA 배너 등장
   - [ ] 오케스트레이터: 선택 모델 파이프라인 실행·하이라이트, 승급 후 CTA 배너
   - [ ] 시뮬레이터: STAGE①에 승급 버전 "✦ 오늘 승급" 칩, 연계 소스 이름 표시
   - [ ] 리본: 4개 탭 어디서든 표시, 클릭 이동, 현재 탭 하이라이트
   - [ ] 백엔드 중지 상태에서도 페이지 정상 렌더(배지·칩만 생략)
