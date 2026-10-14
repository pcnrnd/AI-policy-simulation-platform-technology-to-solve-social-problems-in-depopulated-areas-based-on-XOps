# 외부 dashboard/reporting 모듈 라우팅 런북

이 문서는 **주소 미정 상태의 연결 구조 준비**와 dts 더미 검증 절차다. 실제 파트너 서비스, 운영 도메인·DNS·인증서·AWS 인프라를 구성하거나 연동 완료를 의미하지 않는다. 기존 Caddy+Authelia 앞단 뒤 frontend nginx를 확장하며 backend/DB는 바꾸지 않는다.

## 경로·env 계약

| 변수 | 기본값/형식 | 실제 주소 확정 때 바꿀 값 |
|---|---|---|
| SITE_DOMAIN | 기존 로그인 게이트 도메인, scheme/port/path 없음 | `population.rnd.<보유 도메인>` 등의 확정값; 사용자 담당 |
| AUTH_PUBLIC_URL | `https://<SITE_DOMAIN>/authelia/` | 같은 origin의 확정 portal URL. env 미설정은 기존 값 유지 |
| APP_PUBLIC_PATH | `/rnd/population/` | 우리 앱 별칭. `/`·`/assets/`·`/api/` 호환은 유지 |
| DASHBOARD_PUBLIC_PATH | `${APP_PUBLIC_PATH}dashboard/` | 공개 FE 접두 |
| DASHBOARD_API_PUBLIC_PATH | `${APP_PUBLIC_PATH}dashboard-api/` | 공개 API 접두 |
| DASHBOARD_UPSTREAM | 빈 값, `http(s)://host[:port]` | 파트너 FE origin·port; userinfo/query/fragment/path 금지 |
| DASHBOARD_UPSTREAM_BASE_PATH | `/` | FE upstream의 mount 접두 |
| DASHBOARD_API_UPSTREAM | 빈 값, 같은 origin 형식 | 파트너 API origin·port |
| DASHBOARD_API_UPSTREAM_BASE_PATH | `/` | API upstream의 mount 접두 |
| DASHBOARD_SUPPORTED_MODES | 빈 값, `mock`, `real`, `mock,real` | 파트너가 확인한 지원 모드만 |
| REPORTING_* | dashboard와 동일, 공개 접두 reporting/reporting-api | 리포팅의 실제 값 |

공개 path는 APP_PUBLIC_PATH 아래 유일한 하위 경로이며 끝에 `/`가 있어야 한다. `/api/`, `/assets/`, `/authelia/`, traversal·encoded path 주입·줄바꿈·중복/중첩 접두를 거부한다. env 변수는 frontend 시작 시 검증하고 nginx 설정·`/integration-config.json`을 생성한다. JSON에는 공개 path·설정 여부·지원 모드만 있으며 private upstream/secret을 노출하지 않는다.

미설정 upstream 또는 선택 mode 미지원은 FE/API에서 HTTP **200 준비 중 HTML**과 `X-Integration-State: unconfigured|unsupported_mode`로 응답한다. 이를 데이터 응답으로 파싱하지 않는다. upstream 장애는 502 등 실패 상태를 유지하며 우리 SPA·다른 서비스로 fallback하지 않는다. `/rnd/population/`은 같은 앱 index 별칭이고 내부 탭의 URL router 추가가 아니다.

## 요청·인증·모드

- Caddy가 외부 요청 Remote-User/Groups/Email/Name을 제거한 뒤 Authelia의 성공 identity만 전달한다. frontend는 production에서 host port를 열지 않는다. proxy 네트워크 내부의 직접 호출은 별도 신뢰 경계다.
- module에는 identity와 X-Forwarded-Host/Proto/Prefix를 넘긴다. 앱/Authelia Cookie와 Authorization은 넘기지 않는다. 자체 auth가 필요한 모듈은 별도 계약 전 연결하지 않는다.
- Nav는 FE 설정 시 해당 두 메뉴만 외부 href로 전환하고 `?mode=mock|real`을 붙인다. 미설정이면 기존 내부 탭을 유지한다. 외부 경로의 실패를 내부 탭으로 바꾸는 fallback은 없다. 설정 상태에서 초기 기본 대시보드 탭은 첫 내부 DataOps 탭으로 옮긴다.
- API는 X-Data-Mode 또는 `mode` query를 명시한다. 둘 다 있으면 같아야 한다. 값은 소문자 `mock|real`만 허용하고 중복 mode, 대소문자 다른 mode key, encoded query key, 값 충돌을 400으로 거부한다. 나머지 query와 path percent encoding은 그대로 전달한다.
- FE 최초 접속은 명시한 mode를 사용한다. 생략 시 module별 router-owned mode cookie, 없으면 real을 사용한다. 알 수 없는 cookie 값은 지원 안내로 막는다. 자산 요청의 mode 연속성을 위해 `population_dashboard_mode`/`population_reporting_mode`만 APP_PUBLIC_PATH에 한정해 Secure·HttpOnly·SameSite=Lax로 설정한다. module은 이 Cookie를 받지 않는다.
- 파트너는 FE public base path로 빌드하고 공개 API 접두를 호출한다. 최초 문서의 mode를 모든 데이터 요청에 명시하고 mode 지원이 없으면 real을 대신 요청하지 않는다. 정적 asset에 모드별 데이터가 박혀 있으면 안 된다. deep link와 새로고침은 파트너 자체 index가 책임진다.

## 응답 격리와 한계

nginx는 module의 Set-Cookie·Clear-Site-Data·X-Accel-Redirect·Refresh·Content-Location을 차단한다. X-Accel은 Nginx 내부 redirect 동작 자체를 ignore한다. Location은 확인된 upstream/public prefix 안에서만 치환하며 외부 목적지·경로 탈출은 전달하지 않는다. cache는 private/no-store, Vary는 X-Data-Mode, conditional cache 요청과 ETag/Last-Modified는 전달하지 않는다. HTTPS upstream은 시스템 CA로 검증하며 임의 TLS 검증 해제는 하지 않는다.

같은 origin의 path와 쿠키 제한은 **JavaScript localStorage를 강제 분리하지 못한다**. partner는 module별 key namespace만 사용하고 앱 key·session marker·설정 저장소에 접근하지 않아야 한다. 실제 partner JS의 신뢰·별도 origin 또는 격리 설계가 확인되기 전 실통합은 보류한다. dummy는 앱 저장소를 읽거나 쓰지 않는다. endpoint/SSO/CSP/WebSocket 요구는 [파트너 대조표](../partner-integration-checklist.md)에서 따로 확인한다.

## dts 더미 profile 검증

`compose.integration-test.yaml`은 nginx:alpine의 내부 더미 2종이다. `profiles: [integration-test]`, host port 0, 운영 기본 비활성이다. FE는 base path HTML+JS와 deep link, API는 선택된 identity·mode·raw URI/query·prefix의 text echo만 제공한다. 실제 Cookie/Authorization 값은 echo하지 않는다. 응답 차단 검사용 headers도 포함하지만 데이터·사용자·모델을 만들지 않는다.

운영 `.env`, `users.yml`, `secrets`를 출력하거나 수정하지 않는다. 비밀값 없는 override만 복사한다.

```sh
cd /home/dts/rnd_project/population/population-platform/platform
cp deploy/integration-test.env.example /tmp/population-phase-d.integration.env
# 기존 이미지/설정 검증을 먼저 완료한 뒤, 필요한 서비스만 지정한다.
docker compose --env-file deploy/.env --env-file /tmp/population-phase-d.integration.env \
  -f compose.xops.yaml -f compose.ec2.yaml -f compose.integration-test.yaml \
  --profile integration-test up -d --no-deps integration-dashboard integration-reporting
# frontend 이미지는 코드 후보로 build한다. caddy는 설정 반영에 필요한 경우에만 지정한다.
docker compose --env-file deploy/.env --env-file /tmp/population-phase-d.integration.env \
  -f compose.xops.yaml -f compose.ec2.yaml -f compose.integration-test.yaml \
  --profile integration-test up -d --no-deps --build frontend
docker compose --env-file deploy/.env -f compose.xops.yaml -f compose.ec2.yaml up -d --no-deps caddy
```

writer는 먼저 기본 미설정 경로에서 안내200/fallback0을 검증한다. 이후 **더미 설정 상태를 독립 verifier에게 그대로 남긴다**. verifier는 배포를 변경하지 않는다. Caddy만 필요한 경우 재생성할 수 있으며 Authelia·xops-service·DB는 재생성/재시작하지 않는다. Caddy 이전/이후 같은 로그인 세션으로 200인지 확인한다.

검사: Caddy 2.11.7 validate, nginx -t 기본/설정 상태, Authelia 4.39.28 config validate 기본/설정 public URL(임시 read-only Docker run), env shell 테스트, node 전체, build. 무세션 FE302/API401, 로그인 FE200·asset200·deep link200, 두 모드 echo·query·prefix·identity, 위조 identity 제거, Cookie/Authorization 미전달·응답 headers 차단·mode 모호성400, 루트·API·portal 호환과 보호 StartedAt을 기록한다. 실제 로그인 UI/만료 복귀·Nav ON/OFF 확인은 별도 Neo verifier가 수행한다.

## D 게이트 뒤 기본 복귀(병합 task 담당)

override를 제외해 frontend만 기본 미설정 상태로 재생성한다. `down`, `down -v`, Authelia/DB restart는 사용하지 않는다.

```sh
cd /home/dts/rnd_project/population/population-platform/platform
docker compose --env-file deploy/.env -f compose.xops.yaml -f compose.ec2.yaml \
  up -d --no-deps frontend
docker compose --env-file deploy/.env -f compose.xops.yaml -f compose.ec2.yaml \
  -f compose.integration-test.yaml --profile integration-test stop integration-dashboard integration-reporting
docker compose --env-file deploy/.env -f compose.xops.yaml -f compose.ec2.yaml \
  -f compose.integration-test.yaml --profile integration-test rm -f integration-dashboard integration-reporting
```

기존 운영 .env에 이미 실제 upstream이 있으면 임의로 지우지 말고 중단해 확인한다. 이번 작업은 운영 .env를 변경하지 않는다. 후보 브랜치 삭제 전에 기본/overlay 차이와 게이트 조건을 확인한다.

## 주소 확정 후 담당

사용자 담당: 보유 도메인, DNS/인증서·AWS 설정, 공개 URL·Auth URL 결정. 파트너 담당: 실제 origin/port, FE/API base, SSO/권한, mock/real 지원, storage/cookie, CSP·다운로드 형식. 대조표로 확인한 뒤 보호 컨테이너 변경 허용 범위를 포함한 별도 task에서 실제 연결을 검증한다.

설정 문법 참조: [nginx proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html), [Caddy map](https://caddyserver.com/docs/caddyfile/directives/map), [Authelia templates](https://www.authelia.com/reference/guides/templating/). 버전별 실제 validator 결과를 우선한다.
