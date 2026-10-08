# xops 타 부서 연동 기술문서

> 인구감소 R&D 플랫폼 — 타 부서(`department`, 이름 미정) 대시보드·자동 리포팅이 xops-service API를 읽는 방법
> 대상: `platform/backend/xops-service/` `/api/v3` · `platform/frontend/nginx.conf` · `platform/compose.xops.yaml` · 최종 갱신 2026-10-08
> 함께 전달: [`docs/xops-openapi.json`](xops-openapi.json)(이 커밋 시점 OpenAPI), [`docs/xops-service.md`](xops-service.md), [`docs/deploy/ec2-login-gate.md`](deploy/ec2-login-gate.md)

---

## 1. 개요·책임 경계

같은 도메인에서 nginx 경로로 화면을 나누고, 우리 API 경로 `/api/v3`는 그대로 둔다. 자동 리포팅과 통합 대시보드의 화면·산출물은 타 부서가 만든다. 우리는 그 화면이 읽을 API·인증·nginx 경로·이 문서를 맡는다.

| 우리(PCN) | 타 부서 | 함께 정할 것 |
|---|---|---|
| xops-service API·인증·응답 스키마, 이 문서와 `xops-openapi.json` | 대시보드·자동 리포팅 화면과 산출물, 보고서 문구·서식 | 타 부서 경로 이름(현재 자리표시 `/department/`) |
| `frontend` nginx 라우팅(타 부서 location 포함), compose 생존 확인 | 타 부서 서버(BFF)와 조회 전용 자격증명의 서버 보관 | 서버 간(s2s) 호출 허용 여부 |
| Caddy·Authelia 로그인 게이트 운영 | 자체 upstream 컨테이너 이름·포트 | 조회 전용 자격증명 발급·교체 주기 |
| 우리 UI의 운영(prod) 인증 설계(별도 과제) | | 시드 전용 지표(인구·출산율 등)를 화면에 보일지 |

**선행 조건.** 지금 앱은 `XOPS_ENVIRONMENT=dev`라 토큰 발급이 열려 있다. dev 개방 발급을 닫고 우리 UI의 운영 인증을 바꾸기 전에는 "조회 전용"을 보장하지 못한다(4.3). 그 전에는 이 문서·`xops-openapi.json`·예제를 계약 초안으로만 공유하고 운영 연동 완료로 보지 않는다.

## 2. 연결 구조도

```
브라우저 ─HTTPS─▶ Caddy :443 ─forward_auth─▶ authelia :9091        (/authelia/* 만 로그인 없이 열림)
                     └─▶ frontend :80 (nginx)
                           ├ /api/                      ─▶ xops-service :8000   (/api/v3/... 경로 그대로)
                           ├ = /api/v3/openapi.json     ─▶ xops-service :8000/openapi.json
                           ├ /department/  (자리표시)    ─▶ department-upstream  (타 부서 컨테이너)
                           └ /                          ─▶ 우리 SPA 정적 파일
타 부서 서버(BFF) ─HTTP(같은 Docker 네트워크)─▶ xops-service :8000/api/v3/...   (Authelia 미경유)
                                                   └─▶ pg / timescale / mongo, SQLite
```

- Caddyfile은 `/authelia` 밖의 모든 경로를 `forward_auth` 뒤 `frontend:80`으로 보낸다. 타 부서 화면을 nginx 아래 두면 Caddyfile을 바꾸지 않고 같은 로그인 세션 게이트를 받는다.
- 서버 간 호출은 Docker 네트워크 안에서 `http://xops-service:8000`으로 바로 간다. 로그인 게이트 구성(`compose.ec2.yaml`)에서는 xops-service가 호스트 포트를 열지 않는다(`ports: !reset []`).

## 3. nginx 경로 규칙

`platform/frontend/nginx.conf`

| location | 대상 | 규칙 |
|---|---|---|
| `/api/` | `proxy_pass http://xops-service:8000;` | URI 없는 proxy_pass라 `/api/v3/...`가 그대로 전달된다. **끝에 슬래시를 붙이지 않는다**(붙이면 `/api/` 접두가 잘린다). |
| `= /api/v3/openapi.json` | `proxy_pass http://xops-service:8000/openapi.json;` | FastAPI는 OpenAPI를 루트(`/openapi.json`)에 둔다. 정확 일치가 `/api/` 접두 일치보다 먼저 잡힌다. `/docs`(Swagger UI)는 열지 않는다(CDN 자산을 읽는다). |
| `/department/` | 주석 자리표시 | 타 부서 경로 이름·upstream 이름·포트가 정해지면 주석을 풀고 이름을 바꾼다. |
| `/` | SPA 정적 파일 | `try_files $uri /index.html`. 우리 SPA는 Vite `base`가 없어 `/`에 있어야 한다. |

- 타 부서 경로로 쓰면 안 되는 접두: `/api/`(우리 API), `/assets/`(Vite 산출물), `/authelia/`(로그인 포털).
- 타 부서 location을 켤 때: ① 주석 해제와 이름 지정 ② 타 부서 컨테이너를 같은 compose 네트워크에 붙임 ③ `nginx -t` ④ `frontend` 이미지 재빌드. nginx는 기동 시 upstream 이름을 해석하므로 그 컨테이너가 네트워크에 없으면 frontend가 뜨지 않는다.

## 4. 인증

### 4.1 토큰과 scope

| scope | 받는 방법 | 쓰는 곳 |
|---|---|---|
| `data:read` | 조회 전용 자격증명(`X-Client-Id`·`X-Client-Secret`)으로 `POST /api/v3/dataops/token` | 실데이터 조회(`/realdata/*` GET), 발급 API 목록, `/dataops/{source_id}` GET |
| `data:read data:write` | 주 자격증명(prod) 또는 dev 개방 발급 | 위 전부 + 쓰기(학습 실행·후보 반영·카탈로그 등록·오케스트레이션 쓰기 4개·드리프트 자동 재학습) |

- 토큰은 HS256 JWT이고 유효시간은 3600초다(`exp` 클레임). 만료 전에 다시 발급한다.
- 조회 전용 자격증명은 서버 환경변수 `XOPS_READONLY_CLIENT_ID`·`XOPS_READONLY_CLIENT_SECRET`으로 정한다. dev·prod 모두에서 이 자격증명이면 `data:read`만 준다. 그 id로 틀린 secret을 보내면 dev에서도 401이다. 둘 중 하나만 설정하거나 id가 `XOPS_CLIENT_ID`와 같으면 xops-service가 기동하지 않는다.
- 공개 GET(인증 없음): `/dataops/catalog`, `/overview/summary`, `/orchestration/models·pipelines·runs`, `/monitoring/metrics·drift·explain`, `/realdata/health`. 이 중 오케스트레이션·모니터링 GET은 `data:read` 토큰을 함께 보내면 실데이터 학습 모델·실행을 같은 스키마로 덧붙인다(토큰이 없으면 시드·실측만).

### 4.2 우선 경로 — 타 부서 서버(BFF)

- 타 부서 서버가 조회 전용 자격증명을 **서버에만** 보관하고, 같은 Docker 네트워크에서 `http://xops-service:8000/api/v3/...`를 부른다. 브라우저에는 자격증명과 토큰을 내려보내지 않는다.
- 이 경로는 Authelia 로그인을 거치지 않는다. 그래서 쓰기 경로에 권한 검사가 필요했고(오케스트레이션 쓰기 4개와 드리프트 자동 재학습에 `data:write`), 조회 전용 발급을 따로 두었다.

### 4.3 브라우저 직접 호출의 한계

- 타 부서 화면이 같은 도메인에 있으면 Authelia 세션 쿠키를 공유하므로 따로 로그인할 필요가 없다. 공개 GET은 토큰 없이 부를 수 있다.
- 그러나 같은 origin의 코드는 토큰 발급 경로에도 닿는다. dev에서는 자격증명 없이 `POST /api/v3/dataops/token`이 `data:read data:write` 토큰을 준다. 그러므로 브라우저에서 직접 부르는 방식으로는 조회 전용을 **보장할 수 없다.** 실데이터 조회는 BFF로 받는다.
- 운영(prod) 전환 시 우리 UI도 브라우저에 client secret을 둘 수 없어 토큰 발급이 깨진다. 우리 UI의 운영 인증(사용자·역할 연결)은 이 연동과 분리된 별도 설계 과제다.

## 5. API 목록·요청/응답 예제·오류

### 5.1 타 부서가 읽는 API (`/api/v3`)

| 필요 정보 | 메서드·경로 | 인증 | 응답 형식 |
|---|---|---|---|
| 카탈로그·출처·적재 행수 | `GET /dataops/catalog?live=true` | 공개 | 배열(`SourceSummary`). `is_seed`, `live_rows`(null = 확인 불가) |
| 적재 롤업·모델 F1 | `GET /overview/summary` | 공개 | `OverviewSummary`. 행수는 60초 캐시 |
| 실데이터 모델·활성 버전 | `GET /realdata/models` | `data:read` | `Envelope[list[RealdataModelEntry]]` |
| 후보 버전 | `GET /realdata/models/{model_id}/candidates` | `data:read` | Envelope |
| 평가 | `GET /realdata/models/{model_id}/evaluation?version=` | `data:read` | `Envelope[RealdataEvaluation]` |
| 드리프트·설명 | `GET /realdata/models/{model_id}/drift?version=`, `.../explain?version=&base_ym=&dong_code=` | `data:read` | Envelope |
| 분석·진단·대응 후보 | `GET /realdata/analysis?dong_code=&base_ym=` (region=namwon만) | `data:read` | Envelope(`rules_version`) |
| 학습 실행 이력 | `GET /realdata/training-runs?model_id=` | `data:read` | Envelope. 페이징·기간 조건 없음 |
| 데이터셋 스냅샷 | `GET /realdata/datasets`, `GET /realdata/datasets/{dataset_id}?include_rows=true` | `data:read` | Envelope. rows는 최대 5,000행(넘으면 `message`에 절단 표시) |
| 행정동 대응표 | `GET /realdata/dong-map` | `data:read` | Envelope |
| 상태 확인 | `GET /realdata/health` | 공개 | Envelope(8절) |
| API 계약 | `GET /api/v3/openapi.json`(도메인 경유), `GET /openapi.json`(서버 간) | 게이트/공개 | OpenAPI 3.1 |

실데이터 모델 id는 `namwon-nonlocal-visitors-next-month`(타깃 `nonlocal_visitors`), `namwon-observed-sales-next-month`(타깃 `observed_sales_krw`) 2개로 고정이다.

### 5.2 서버 간(BFF) 요청 예제

타 부서 서버(같은 Docker 네트워크)에서 실행한다. 응답의 값은 예시다.

```sh
X=http://xops-service:8000
RO_ID='<조회 전용 client id>'          # 서버 환경변수 XOPS_READONLY_CLIENT_ID 값
RO_SECRET='<조회 전용 client secret>'  # 서버 환경변수 XOPS_READONLY_CLIENT_SECRET 값

# 1) 조회 전용 토큰 발급 → 200 {"access_token":"eyJ...","token_type":"Bearer","scope":"data:read"}
TOKEN=$(curl -s -X POST "$X/api/v3/dataops/token" -H "X-Client-Id: $RO_ID" -H "X-Client-Secret: $RO_SECRET" \
  | sed -E 's/.*"access_token":"([^"]+)".*/\1/')

# 2) 공개 조회
curl -s "$X/api/v3/overview/summary"          # 200 {"status":200,...,"source_kind":"database","sources":[...],"model":...}
curl -s "$X/api/v3/dataops/catalog?live=true"  # 200 [{"id":"ds_08_admin_boundary",...,"is_seed":false,"live_rows":251},...]
curl -s "$X/api/v3/realdata/health"           # 200 {"status":"ok","data":{"pg_dsn_configured":true,"table_status":{...}},...}

# 3) 실데이터 조회(data:read)
curl -s -H "Authorization: Bearer $TOKEN" "$X/api/v3/realdata/models"
# 200 {"status":"ok","message":null,"data":[{"model_id":"namwon-nonlocal-visitors-next-month",
#      "target":"nonlocal_visitors","active_version":"v20260918-...","applied_at":"2026-09-18T...+00:00",
#      "previous_version":null,"retrain_needed":false},...],"provenance":{...,"computed_at":"..."}}
curl -s -H "Authorization: Bearer $TOKEN" "$X/api/v3/realdata/models/namwon-observed-sales-next-month/evaluation"
# 200 {"status":"ok","data":{"validation":{"kind":"validation","metrics":{"mae":...,"rmse":...,"wape":...},
#      "baseline":{"name":"yoy_or_prev_month",...},"eval_period":{"from":202506,"to":202508,"n":69}},
#      "operational":{"kind":"pending","pending_months":[202510]}},"provenance":{...,"version":"v..."}}
#   활성 모델이 없으면 200 {"status":"model_required","message":"활성 모델이 없습니다.","data":null,...}
curl -s -H "Authorization: Bearer $TOKEN" "$X/api/v3/realdata/training-runs"
curl -s -H "Authorization: Bearer $TOKEN" "$X/api/v3/realdata/datasets"

# 4) 권한 경계(현행 401 규칙)
curl -s -o /dev/null -w '%{http_code}\n' "$X/api/v3/realdata/models"                       # 401 토큰 없음
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"model_id":"namwon-observed-sales-next-month","dataset_id":"ds-0123456789ab"}' \
  "$X/api/v3/realdata/training-runs"                                                       # 401 scope 'data:write' 없음
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Content-Type: application/json' \
  -d '{"model_id":"population-forecast","trigger":"manual"}' "$X/api/v3/orchestration/events"  # 401 토큰 없음
```

### 5.3 브라우저 경로(로그인 게이트 경유) 예제

로그인 세션 쿠키가 있어야 한다. Caddy 자체 CA(검증 환경)면 `-k`를 붙이고, DNS가 없으면 `--resolve <SITE_DOMAIN>:443:<서버 IP>`를 붙인다.

```sh
B=https://<SITE_DOMAIN>
curl -s -o /dev/null -w '%{http_code}\n' "$B/api/v3/overview/summary"     # 302 또는 401 — 세션 없음
curl -s -c jar -H 'Content-Type: application/json' \
  -d '{"username":"<ID>","password":"<PW>","keepMeLoggedIn":false}' "$B/authelia/api/firstfactor"
curl -s -b jar "$B/api/v3/overview/summary"                               # 200, source_kind 포함
curl -s -b jar "$B/api/v3/dataops/catalog"                                # 200, is_seed 포함
curl -s -b jar -H "Authorization: Bearer $TOKEN" "$B/api/v3/realdata/models"  # 200 {"status":"ok",...}
curl -s -b jar "$B/api/v3/openapi.json"                                   # 200 OpenAPI(= docs/xops-openapi.json)
```

### 5.4 오류·상태 매핑

HTTP 코드는 요청·인증·서버 오류에만 쓴다. 데이터가 없거나 모자란 상태는 HTTP 200 본문의 상태 필드로 알린다.

| 상황 | HTTP | 본문 |
|---|---|---|
| 토큰 없음·서명 불일치·만료 | 401 | `{"status":401,"error":"AuthError","message":"JWT 토큰이 필요합니다. ..."}` |
| scope 부족(조회 전용 토큰으로 쓰기) | **401**(현행) | `{"status":401,"error":"AuthError","message":"scope 'data:write' 권한이 없습니다."}` |
| 발급 자격증명 불일치(prod, 또는 조회 전용 id의 틀린 secret) | 401 | `{"status":401,"error":"AuthError","message":"... 자격증명이 유효하지 않습니다 ..."}` |
| 로그인 세션 없음(도메인 경유) | 302(포털로) 또는 401 | Authelia 응답 — 앱까지 가지 않는다 |
| 카탈로그·실행 이력에 없는 id | 404 | `{"status":404,"error":"SourceNotFoundError","message":...}` |
| 실데이터의 알 수 없는 model_id·job | 404 | `{"detail":"알 수 없는 model_id: ..."}` |
| 요청 값 검증 실패 | 422 | `{"detail":[{"type":"missing","loc":["query","base_ym"],...}]}` |
| 같은 모델 학습 중복 실행 | 409 | `{"detail":"..."}` |
| 후보 반영 조건 미달 | 400 | `{"detail":{"reasons":[...]}}` |
| 실데이터 업무 상태 | 200 | Envelope `status`: `ok`·`empty`·`insufficient_data`·`model_required`·`pending`·`error` |
| 모니터링 값의 출처 | 200 | `source`: `measured`(실측)·`seed`(시드)·`realdata`·`null`(값 없음) |
| DataOps 저장소 상태 | 200 | `source_kind`: `database`(실제로 셈)·`in-memory`(확인 못 함·스텁) |

- 권한 부족을 403으로 나누는 것은 권고 사항이며 계약 변경이라 별도로 진행한다. 지금은 미인증과 권한 부족이 모두 401이다.
- Envelope `status:"error"`도 HTTP 200이다(예: `/realdata/health`의 DB 장애). 판정은 본문으로 한다.
- 평가 지표 `wape`는 Σ|오차| / Σ|관측| 비율값이라 1을 넘을 수 있다(백분율 표시는 ×100).

## 6. 시드 데이터 구분

데모용 시드는 지우지 않고 목록에서만 뺀다. 타 부서 화면은 기본값(`include_seed` 생략 = false)으로 부른다.

| 표시 | 위치 | 뜻 |
|---|---|---|
| `is_seed` | `/dataops/catalog` 항목 | `true`면 데모 시드 소스(ds_01~07). 실데이터는 ds_08~13 |
| `source` | `/monitoring/metrics·drift·explain` | `seed`면 시드 값. 기본값에서는 시드 대신 `null`(빈 응답) |
| `metrics_source` | `/orchestration/models`, `/overview/summary`의 `model` | `seed`면 지표가 시드 |
| `include_seed` | 쿼리 파라미터 | 기본 false. `true`는 우리 화면의 데모 표시 ON 전용 |

- `include_seed`가 기본 false인 목록: `/dataops/catalog`, `/overview/summary`, `/orchestration/models·pipelines·runs`, `/monitoring/metrics·drift·explain`. 생략하면 시드 항목이 없다.
- 단건(`/dataops/catalog/{id}`, `/orchestration/runs/{id}`·`/logs`)과 쓰기 경로는 이 필터 대상이 아니다. 그 밖의 endpoint는 시드 제외를 보장하지 않는다.
- `/realdata/*`는 실데이터 저장소(rd_*)만 읽는다. 지역 인구·출산율·소멸위험지수는 원천이 없어 API로 제공하지 않는다(우리 화면의 해당 값은 시드다).

## 7. 리포트용 호출 순서

우리 리포트 화면(`platform/frontend/src/lib/dataopsApi.js`의 `fetchReportData`)이 쓰는 순서다. 타 부서 리포트도 같은 순서로 재현할 수 있다.

1. 토큰 발급(`data:read`).
2. `GET /realdata/models` → `data[]`에서 `active_version`이 있는 첫 모델을 고른다. 없으면 리포트 지표는 빈 상태다.
3. 아래를 동시에 부른다(`version` = 그 모델의 `active_version`).
   - `GET /realdata/models/{model_id}/evaluation?version=`
   - `GET /realdata/models/{model_id}/drift?version=`
   - 설명(SHAP) 바인딩: ① `GET /realdata/datasets` → `data.datasets[]`에서 `spec.model_id`가 같은 첫 항목(생성 시각 내림차순이라 최신) ② `GET /realdata/datasets/{dataset_id}?include_rows=true` → `message`가 비어 있을 때만(절단 없음) `base_ym == observed_to`인 행 중 `local_visitors`가 가장 큰 `dong_code` ③ `GET /realdata/models/{model_id}/explain?version=&base_ym=<observed_to>&dong_code=` 와 `GET /realdata/dong-map`(행정동 이름)
4. 설명 바인딩의 어느 단계든 실패하면 설명만 빈 상태로 두고 평가·드리프트는 그대로 쓴다.

집약 API(`GET /realdata/report-context?model_id=`)는 타 부서가 이 순서를 재구현하기 어렵다고 할 때만 추가한다(기존 계산 재사용, 신규 산식 없음).

## 8. 상태 확인

| 확인 | 경로 | 판정 |
|---|---|---|
| 데이터 준비 | `GET /api/v3/realdata/health`(공개) | **항상 HTTP 200**. `status:"ok"`면 PG 연결 성공, `data.table_status`가 허용 테이블별 존재 여부. `status:"error"`면 DSN 미설정 또는 PG 장애(`message`). PG만 본다(timescale·mongo는 보지 않는다) |
| 프로세스 생존 | `GET /` → `{"xops":"connected"}` | compose healthcheck가 컨테이너 안에서 부른다(`docker compose ps`의 `healthy`). 생존만 뜻하며 DB 준비가 아니다. 도메인의 `/`는 SPA라 이 경로로는 닿지 않는다 |

- readiness(비-200 응답이나 health `status` 기반 프로브)는 별도 제안으로 남긴다.

## 9. 검증 체크리스트

배포마다 아래를 확인한다. dts 결과는 `F:\etc\review-system\artifacts\active\rnd-platform-dts-20261008\P\`(P6)에 있다.

1. 로그인 세션으로 `GET /api/v3/overview/summary`, `/api/v3/dataops/catalog`가 200이고 `source_kind`·`is_seed`가 있다. 세션이 없으면 302 또는 401이다.
2. 세션과 `Authorization: Bearer`를 함께 보낸 `GET /api/v3/realdata/models`가 200이고 `status:"ok"`다(Authelia를 통과한 Bearer 헤더가 앱까지 간다).
3. 조회 전용 토큰은 `GET /realdata/*`에서 200, `POST /realdata/training-runs`에서 401이다. 토큰 없는 `POST /orchestration/events`는 401이다.
4. 타 부서 컨테이너(같은 Docker 네트워크)에서 `http://xops-service:8000/api/v3/realdata/health`가 HTTP 200이다. PG 상태는 HTTP 코드가 아니라 `status`로 판정한다.
5. `include_seed`를 생략한 6절 목록 호출에 시드 항목이 없다(`is_seed:true`, `source:"seed"`, `metrics_source:"seed"` 없음).
6. `GET /api/v3/openapi.json`(도메인 경유)이 이 커밋의 `docs/xops-openapi.json`과 같다. 단 FastAPI가 소유하는 422 스키마(`ValidationError`·`HTTPValidationError`)는 설치된 FastAPI 버전에 따라 선택 필드가 다를 수 있다(2026-10-08 dts 이미지의 0.142.4는 `input`·`ctx`를 더한다). `requirements.txt`가 하한만 고정하기 때문이며, 우리 경로·스키마는 같아야 한다.

`docs/xops-openapi.json` 다시 만들기(API를 바꾼 커밋마다 — `test_response_contracts.py::test_openapi_snapshot_is_current`가 어긋나면 실패한다). 앱이 기동 로그를 표준출력에 쓰므로 리다이렉트하지 않고 파일에 직접 쓴다.

```sh
cd platform/backend/xops-service
python -c "import json; from main import app; open('../../../docs/xops-openapi.json', 'w', encoding='utf-8', newline='\n').write(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + '\n')"
```
