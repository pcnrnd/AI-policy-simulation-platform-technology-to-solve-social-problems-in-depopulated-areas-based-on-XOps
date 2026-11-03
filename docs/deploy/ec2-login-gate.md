# EC2 로그인 게이트 런북 (Caddy + Authelia)

플랫폼을 외부에 공개할 때 앞단에 로그인 화면을 붙여, 로그인한 계정만 화면과 `/api`에 닿게 한다.
기존 로그인 게이트는 `platform/compose.ec2.yaml`을 겹쳐 쓴다. Phase D는 그 뒤 frontend nginx의 외부 모듈 경로를 추가하며 Caddy+Authelia는 유지한다. 주소 미정의 환경변수·더미 profile·안전한 단위 배포는 [외부 모듈 런북](external-modules.md)을 따른다. 실제 파트너 연결은 미완료다.
저장소 3종(pg·timescale·mongo)은 앱 project `platform`과 분리된 project `population-db`로 따로 띄운다. 앱 Compose에는 DB가 없으며, 운영 세부는 [DB 분리 운영 절차](../../platform/deploy/db-separation-runbook.md)를 따른다.

```
브라우저 ──HTTPS──▶ Caddy :443 ──(forward_auth)──▶ Authelia :9091  ← 세션 없으면 로그인 포털로
                      │
                      └──▶ frontend(nginx) :80 ──/api──▶ xops-service :8000 ──(망 population-db-data)──▶ pg / timescale / mongo
                           └──────────────── project platform ─────────────────┘                        └─ project population-db ─┘
```

| 파일 | 역할 |
|---|---|
| `platform/compose.ec2.yaml` | Caddy·Authelia 추가, 앱 포트 비공개, 비밀값 주입 |
| `platform/compose.db.yaml` | DB 3종 전용 Compose(project `population-db`). 볼륨·망은 외부 소유(external)라 없으면 기동 실패 |
| `platform/compose.db.ec2.yaml` | 운영 DB 덮어쓰기: DB 비밀번호, 포트 127.0.0.1 한정, 이미지 RepoDigest 고정(`pull_policy: never`) |
| `platform/deploy/db.volumes.env.example` | DB가 붙을 실볼륨 이름·이미지 목록 (서버에서 `db.volumes.env`로 복사) |
| `platform/deploy/db-separation-runbook.md` | DB 분리 전환·복귀·금지 명령 |
| `platform/deploy/caddy/Caddyfile` | HTTPS 종단, `/authelia` 포털 프록시, 나머지 전부 `forward_auth` 보호 |
| `platform/deploy/authelia/configuration.yml` | 파일 사용자(argon2id), 기본 deny + 도메인 전체 one_factor, IP 차단 규칙 |
| `platform/deploy/authelia/users.example.yml` | 계정 형식 예시 (실제 `users.yml`은 커밋 금지) |
| `platform/deploy/.env.example` | 필요한 환경변수 목록 (실제 `.env`는 커밋 금지) |

- 로그인 포털: `https://<SITE_DOMAIN>/authelia/`
- `AUTH_PUBLIC_URL`은 같은 origin 포털 URL의 선택 env다. 비어 있으면 Caddy·Authelia 모두 기존 SITE_DOMAIN 기반 URL을 사용한다. 하위 앱 별칭 `/rnd/population/`·외부 모듈도 동일 게이트와 `/authelia/?rd=<전체 URL>` 복귀를 받는다. domain·DNS·인증서 확정 및 변경은 사용자 담당 후속이다.
- 앱은 `XOPS_ENVIRONMENT=dev` 그대로다. 앱의 토큰 발급은 열려 있지만 Caddy 앞단 로그인을 통과한 요청만 닿는다.
- 호스트에 열리는 포트는 Caddy의 80·443뿐이다. 8000·8080은 바인딩하지 않고, DB(5433·5434·27017)는 `127.0.0.1`에만 연다.

## 1. 사전 조건

- Docker Engine과 **Docker Compose 2.24 이상**(`!override`/`!reset` 태그). `docker compose version`으로 확인한다.
- 도메인 A 레코드가 EC2 공인 IP(탄력적 IP 권장)를 가리킨다.
- 보안그룹 인바운드
  - **80/tcp: 0.0.0.0/0 전체 공개.** Let's Encrypt가 인증서 발급·갱신(HTTP-01)을 위해 불특정 IP에서 접속하고, HTTP→HTTPS 리다이렉트도 80에서 한다. 관리자 IP로 좁히면 인증서 갱신이 실패한다.
  - 443/tcp: 0.0.0.0/0
  - 22/tcp: 관리자 IP만
  - 5433·5434·27017·8000·8080·9091은 열지 않는다.

## 2. 최초 설정 (서버에서 1회)

저장소 루트에서 실행한다.

```sh
cp platform/deploy/.env.example platform/deploy/.env
chmod 600 platform/deploy/.env
# 편집: SITE_DOMAIN, CADDY_TLS(인증서 알림 이메일), XOPS_JWT_SECRET, DB_PASSWORD
#   XOPS_JWT_SECRET=$(openssl rand -hex 32), DB_PASSWORD=$(openssl rand -hex 24) 처럼 만든다.
#   CADDY_HTTP_PORT·CADDY_HTTPS_PORT는 운영에서 비워 둔다.

mkdir -p platform/deploy/authelia/secrets
( umask 077
  openssl rand -hex 64 > platform/deploy/authelia/secrets/session_secret
  openssl rand -hex 64 > platform/deploy/authelia/secrets/storage_encryption_key )
```

- `storage_encryption_key`를 바꾸거나 잃으면 Authelia SQLite를 읽지 못한다. `.env`와 함께 서버 밖에 안전하게 보관한다.
- `DB_PASSWORD`는 DB 볼륨을 **처음 만들 때만** 적용된다. 기존 볼륨이 있는 서버에서 바꾸면 앱이 DB에 접속하지 못한다.
- 첫 기동 전에 `platform/deploy/authelia/users.yml`에 계정을 최소 1개 만든다(3절). 파일이 없으면 Authelia가 기동하지 않는다.

DB 준비(새 서버에서 1회). DB Compose는 볼륨·망을 만들지 않으므로 먼저 만들고 그 이름을 `db.volumes.env`에 적는다.
이미 DB 볼륨이 있는 서버(dts 등)는 새로 만들지 말고 `docker inspect`로 확인한 기존 이름을 쓴다.

```sh
docker network create --driver bridge population-db-data   # 이미 있으면 driver·internal=false만 확인
docker volume create platform_pg-data
docker volume create platform_timescale-data
docker volume create platform_mongo-data
docker volume create population-db_mongo-config
cp platform/deploy/db.volumes.env.example platform/deploy/db.volumes.env
# 편집: POP_DB_*_VOLUME을 위 이름으로, POP_DB_*_IMAGE는 이 서버에 pull해 둔 이미지의 RepoDigest로
#   docker pull postgis/postgis:16-3.4 timescale/timescaledb:latest-pg16 mongo:7
#   docker image inspect --format '{{index .RepoDigests 0}}' <이미지>
```

- 망은 `internal=false`여야 한다. internal 망에 붙은 컨테이너는 127.0.0.1 포트도 호스트에 공개되지 않는다.
- 새 빈 볼륨이면 처음 기동할 때 `platform/initdb/`의 초기화 스크립트가 실행된다. 기존 볼륨에서는 실행되지 않는다.

## 3. 계정 관리

### 발급

1. 비밀번호 해시를 만든다. 아래 둘 중 하나를 쓴다(비밀번호를 명령줄에 쓰지 않아 셸 기록에 남지 않는다).

   ```sh
   # 직접 정한 비밀번호 — 프롬프트로 두 번 입력
   docker run --rm -it authelia/authelia:4.39.28 authelia crypto hash generate argon2
   # 임의 비밀번호 생성 — 출력된 Random Password를 사용자에게 안전한 경로로 전달
   docker run --rm authelia/authelia:4.39.28 authelia crypto hash generate argon2 --random --random.length 20
   ```

   출력의 `Digest: $argon2id$v=19$...` 문자열이 해시다.

2. `platform/deploy/authelia/users.yml`에 항목을 추가한다. 형식은 `users.example.yml`을 따른다.

   ```yaml
   users:
     hong:
       disabled: false
       displayname: '홍길동'
       password: '$argon2id$v=19$m=65536,t=3,p=4$...'
       email: 'hong@example.com'
       groups: []
   ```

3. `watch: true`라 파일을 저장하면 다시 읽는다. 반영되지 않으면 `$DC restart authelia`를 한다(아래 5절의 `$DC`). 재시작하면 세션 저장소가 메모리라 모든 사용자가 로그아웃된다.

웹에서 비밀번호 재설정·변경은 막아 두었다(메일 없이 운영). 비밀번호를 바꿀 때도 관리자가 해시를 새로 만들어 교체한다.

### 만료

- 해당 계정을 `disabled: true`로 바꾸거나 항목을 지운다. 이후 그 계정으로는 로그인할 수 없다.
- 이미 로그인한 세션까지 즉시 끊으려면 `$DC restart authelia`로 전체 세션을 비운다.

### 차단 해제

같은 IP에서 10분 안에 로그인을 5회 실패하면 그 IP가 30분 동안 차단된다(`regulation`, `modes: [ip]`).

```sh
$DC exec authelia authelia storage bans ip list            # 차단 목록과 ID
$DC exec authelia authelia storage bans ip revoke <IP>      # 해당 IP 차단 해제
$DC exec authelia authelia storage bans ip revoke --id <ID> # 또는 ID로 해제
```

Caddy가 클라이언트의 `X-Forwarded-For`를 덮어써 Authelia에 실제 접속 IP를 넘긴다. 나중에 ALB·CloudFront를 앞에 두면
Caddyfile에 `trusted_proxies`를 설정해야 IP 차단이 프록시 IP 하나에 몰리지 않는다.

**배포 직후 1회 확인:** 로컬 검증에서는 docker-proxy를 거쳐 모든 요청이 도커 게이트웨이 IP(`172.18.0.1`)로 보였다.
서버에서 외부 PC로 일부러 한 번 틀린 뒤 `$DC logs authelia | grep remote_ip`에 그 PC의 공인 IP가 찍히는지 본다.
게이트웨이 IP로 찍히면 한 사람의 5회 실패가 모든 사용자를 30분 막으므로 공개 전에 원인을 해결한다.

## 4. 로컬 검증 방법

시스템 hosts 파일은 건드리지 않는다. `*.localtest.me`는 공개 DNS에서 127.0.0.1(::1)로 풀리므로 그대로 쓰고,
Caddy는 `tls internal`(자체 CA)로 인증서를 만든다.

1. 2절과 같이 `.env`·`secrets/`·`users.yml`을 만들되 `.env`는 다음처럼 둔다.

   ```sh
   SITE_DOMAIN=pop.localtest.me
   CADDY_TLS=internal
   CADDY_HTTP_PORT=18880   # 호스트 80이 점유된 경우만
   CADDY_HTTPS_PORT=443    # 호스트 443이 점유됐으면 다른 포트로 두고 curl --connect-to 사용
   XOPS_JWT_SECRET=...
   DB_PASSWORD=...
   ```

2. 프로젝트 이름을 따로 줘서 기존 로컬 스택의 컨테이너·볼륨과 분리해 띄운다. 5433·5434·27017·443을 쓰는 컨테이너가 이미 있으면 멈추지 말고 중단한다.
   DB는 테스트 전용 볼륨으로 먼저 띄운다. 망 이름 `population-db-data`는 고정이라, 이 PC에 이미 그 망을 쓰는 DB가 떠 있으면 중단한다.

   ```sh
   docker network create --driver bridge population-db-data
   for v in pg timescale mongo mongo-config; do docker volume create logintest_$v; done
   # platform/deploy/db.volumes.env.logintest: POP_DB_PG_VOLUME=logintest_pg, POP_DB_TIMESCALE_VOLUME=logintest_timescale,
   #   POP_DB_MONGO_VOLUME=logintest_mongo, POP_DB_MONGO_CONFIG_VOLUME=logintest_mongo-config, POP_DB_*_IMAGE=로컬 RepoDigest
   LTDB="docker compose -p popdecline-logintest-db --env-file platform/deploy/.env --env-file platform/deploy/db.volumes.env.logintest -f platform/compose.db.yaml -f platform/compose.db.ec2.yaml"
   LT="docker compose --env-file platform/deploy/.env -p popdecline-logintest -f platform/compose.xops.yaml -f platform/compose.ec2.yaml"
   $LTDB up -d     # healthy 확인 후
   $LT up -d --build
   ```

3. 확인한다(`-k`는 자체 CA 때문).

   ```sh
   B=https://pop.localtest.me
   curl -sk -o /dev/null -D - $B/                          # 302, location: $B/authelia/?rd=...
   curl -sk -o /dev/null -D - $B/api/v3/realdata/health    # 302 또는 401, 데이터 없음
   curl -sk -o /dev/null -w '%{http_code}\n' -H 'Accept: application/json' $B/api/v3/realdata/health   # 401
   # 로그인 → 세션 쿠키 저장
   curl -sk -c jar -H 'Content-Type: application/json' \
        -d '{"username":"<ID>","password":"<PW>","keepMeLoggedIn":false}' $B/authelia/api/firstfactor
   curl -sk -b jar -o /dev/null -w '%{http_code}\n' $B/                          # 200
   curl -sk -b jar $B/api/v3/realdata/health                                     # 200 + JSON
   # 틀린 비밀번호 5회 → 이후 올바른 비밀번호도 거부(차단)
   docker ps --format '{{.Names}}\t{{.Ports}}'             # 8000·8080 없음, DB는 127.0.0.1만
   ```

   브라우저로도 `https://pop.localtest.me/`에 접속하면 로그인 화면이 뜬다(자체 CA 경고는 무시).

4. 정리: 테스트 프로젝트와 테스트 볼륨만 지운다. 다른 컨테이너·볼륨·이미지는 건드리지 않는다(이 절차는 로컬 테스트 전용이며 서버에서는 쓰지 않는다).

   ```sh
   $LT down -v
   $LTDB down
   for v in pg timescale mongo mongo-config; do docker volume rm logintest_$v; done
   docker network rm population-db-data   # 이 테스트에서 만든 경우만
   ```

## 5. 서버 기동

DB와 앱을 각각의 project로 다룬다. 기동 순서는 DB → 앱이다(앱은 DB를 depends_on으로 기다리지 않는다).

```sh
DC="docker compose -p platform --env-file platform/deploy/.env -f platform/compose.xops.yaml -f platform/compose.ec2.yaml"
DB="docker compose -p population-db --env-file platform/deploy/.env --env-file platform/deploy/db.volumes.env -f platform/compose.db.yaml -f platform/compose.db.ec2.yaml"

$DB config --quiet && $DB up -d   # DB 기동 — 세 컨테이너 healthy 확인
$DC up -d --build                 # 앱 기동·갱신(DB는 건드리지 않는다)
$DC ps; $DB ps                    # 상태
$DC logs -f caddy authelia
```

- 앱만 바꿀 때는 `$DC up -d --no-deps <서비스>`만 쓰고 DB Compose를 읽거나 DB를 재시작하지 않는다.
- 서버에서는 어떤 project든 `down`, `-v`, `--remove-orphans`, `prune`을 쓰지 않는다. DB, 로그인 이력·차단 목록(`authelia-data`), 인증서(`caddy-data`)가 지워질 수 있다. 앱을 멈출 때는 `$DC stop`을 쓴다.
- 분리 직후 `platform` project에 이전 DB 컨테이너(`platform-pg-1` 등)가 정지 상태로 남아 orphan 경고가 뜬다. 복귀용으로 보존 중이므로 경고는 무시한다.
- xops-service를 다시 만든 뒤에는 `docker restart platform-frontend-1`로 nginx가 API 주소를 다시 찾게 한다.
- 실데이터 적재 스크립트는 DB 비밀번호가 바뀌었으므로 DSN을 넘긴다.
  `python platform/scripts/load_real_data.py --pg-dsn postgresql://xops:<DB_PASSWORD>@127.0.0.1:5433/xops_dataops`

## 6. 알아둘 점

- **알림(notifier)은 filesystem이다.** 공식 문서상 이 방식은 테스트용이며 운영에서는 권장하지 않는다.
  메일 없이 운영하려고 택했고, 비밀번호 재설정·변경을 막아 두어 운영 흐름에서는 알림이 쓰이지 않는다.
  사용자 셀프 재설정이 필요해지면 SMTP notifier로 바꾼다.
- 세션은 무활동 1시간, 절대 12시간(`configuration.yml`의 `session.inactivity`·`expiration`)이다. 로그인 화면의 "Remember me"를 고르면 더 길게 유지된다.
- 인증은 1단계(비밀번호)다. 2단계(TOTP 등)가 필요하면 `access_control`의 `one_factor`를 `two_factor`로 바꾸고 SMTP를 붙인다.
- Authelia 기동 로그의 `chown: /config...: Read-only file system`은 설정 폴더를 읽기 전용으로 마운트해서 나는 경고다. 동작에는 영향이 없다.

### 세션 정책·화면 변경 반영

```sh
$DC up -d --no-deps --force-recreate authelia   # 세션 정책(configuration.yml) 반영. 세션 저장소가 메모리라 전원 로그아웃된다
$DC up -d --no-deps --build frontend            # 화면 변경 반영
```

- `--no-deps`가 없으면 depends_on 때문에 frontend 갱신 시 xops-service까지 재빌드·재생성된다.

- 세션이 끊기면 화면이 로그인 화면으로 보내고, 로그인 뒤 원래 화면으로 돌아와 "다시 로그인했습니다" 안내가 뜬다.

## 7. 데모 운영 이력(`XOPS_DEMO_HISTORY`) 켜기

데모 표시 ON(mock 모드) 화면이 운영 중인 서버처럼 보이도록, 새 설치에서 한 번 과거 12주 운영 이력을 만든다
(모델 4개·학습 12회·활성화 12건·복원 1건·소유 감사 37건, DataOps 사용자 소스 2건·발급 API 3건).
기본은 꺼짐(0)이며 실데이터 모드와 모드 헤더 없는 호출에는 이 기록이 보이지 않는다. 앱 기동 시
`xops-data` 볼륨의 파일 잠금 안에서 동기로 실행되고 상한은 60초다. 상한을 넘기면 쓰기를 멈추고 다음 기동 때 잇는다.
상한 60초는 xops-service 헬스체크가 unhealthy로 바꾸기까지의 약 90초(start_period 20초 + 15초 간격 5회)보다 짧다.

### 7.1 배포 전 확인(읽기 전용) — 하나라도 다르면 중단하고 보고한다

```sh
docker exec platform-xops-service-1 python -c "import sqlite3; c=sqlite3.connect('file:/app/data/xops.db?mode=ro', uri=True); print('mock_rows', c.execute(\"select count(*) from mo_assets where mode='mock'\").fetchone()[0]); print('state_table', c.execute(\"select count(*) from sqlite_master where name='ops_history_state'\").fetchone()[0])"
```

- `mock_rows 0`, `state_table 0`이어야 새 설치다. mock 기록이 있으면 켜도 건너뛰므로(덮어쓰지 않음) 켤 이유가 없다.
- 실데이터 기준선(대표 조회의 `source_kind=database`·건수)을 같은 시점에 기록해 둔다.

### 7.2 백업 — 앱만 멈추고 xops-data 볼륨 전체를 보존한다

```sh
TS=$(date +%Y%m%d-%H%M%S); BK=/home/<계정>/backups/population-demo-history-$TS; mkdir -p $BK
docker image inspect platform-xops-service --format '{{.Id}}' > $BK/xops-service.image   # 복귀용 현재 이미지
docker image inspect platform-frontend --format '{{.Id}}' > $BK/frontend.image
$DC stop xops-service                                        # 쓰기 중지(DB 3종은 건드리지 않는다)
docker run --rm -v platform_xops-data:/data:ro -v $BK:/backup alpine tar czf /backup/xops-data.tgz -C /data .   # SQLite WAL 포함
cp platform/deploy/.env $BK/env.bak
sha256sum $BK/* > $BK/SHA256SUMS
```

### 7.3 반영 — backend → 이력 확인 → frontend 순서

```sh
git -C <checkout> checkout <병합 SHA>
# 플래그 행만 1로 바꾸거나(기존 0 포함) 없으면 추가한다 — 다른 값은 그대로
if grep -q '^XOPS_DEMO_HISTORY=' platform/deploy/.env; then sed -i 's/^XOPS_DEMO_HISTORY=.*/XOPS_DEMO_HISTORY=1/' platform/deploy/.env; else echo 'XOPS_DEMO_HISTORY=1' >> platform/deploy/.env; fi
diff <(grep -v '^XOPS_DEMO_HISTORY=' $BK/env.bak) <(grep -v '^XOPS_DEMO_HISTORY=' platform/deploy/.env) && echo env_other_values_same
$DC config | grep XOPS_DEMO_HISTORY                          # 'XOPS_DEMO_HISTORY: "1"'가 아니면 여기서 중단(재생성하지 않는다)
$DC up -d --no-deps --build xops-service
$DC logs xops-service | grep '운영 이력 재구성'               # '운영 이력 재구성: completed'
docker exec platform-xops-service-1 python -c "from src.modelops import demo_history as h; s=h.read_state(); print(s['status'], s['manifest_hash']==h.manifest_hash(), h.verify(s['base_time']))"
```

- 성공 판정은 healthy가 아니라 위 확인값이다: `completed True`, `audit_total 37`, `pointers` 4개 모두 `ok`(남원 생활인구는 2차 활성·3차 이전),
  `user_sources 2`, `built_apis 3`, `problems []`.
- 그다음 `$DC up -d --no-deps --build frontend` → `docker restart platform-frontend-1`(nginx가 API 주소를 다시 찾게 한다).
- `down`·`-v`·`--remove-orphans`·`prune` 금지, DB 3종(project `population-db`)은 읽지도 재시작하지도 않는다.

### 7.4 실패 시 복귀

헤더 없는 요청을 실데이터로 보는 누출 차단은 이 변경과 함께 들어왔다. **mock 기록이 하나라도 저장된 뒤 backend를 이
변경 이전 SHA로 되돌리면** 헤더 없는 `/dataops/catalog`·`/dataops/apis` 등에 새 mock 사용자 소스·발급 API가 다시 보인다.
backend 재생성(7.3) 자체가 쓰기를 시작하므로, 분기는 확인 단계가 아니라 **볼륨에 실제로 쓰였는지**로 고른다.

먼저 backend를 멈춘 상태에서(이전 backend를 띄우기 전에) 볼륨을 읽기 전용으로 확인한다.

```sh
$DC stop xops-service
docker run --rm -v platform_xops-data:/data:ro platform-xops-service python -c "import sqlite3; c=sqlite3.connect('file:/data/xops.db?mode=ro', uri=True); q=lambda s: c.execute(s).fetchone()[0]; t=lambda n: q(f\"select count(*) from sqlite_master where name='{n}'\"); print('mock_rows', q(\"select count(*) from mo_assets where mode='mock'\")); print('state_table', t('ops_history_state')); print('mock_sources', q(\"select count(*) from user_sources where schema_json like '%\\\"mode\\\": \\\"mock\\\"%'\")); print('mock_apis', q(\"select count(*) from built_apis where definition_json like '%\\\"mode\\\": \\\"mock\\\"%'\"))"
```

1. **네 값이 모두 0**(아무것도 쓰이지 않음): `.env`의 `XOPS_DEMO_HISTORY=0` → `git checkout <이전 SHA>` →
   `$DC up -d --no-deps --build xops-service` → `$DC up -d --no-deps --build frontend` → `docker restart platform-frontend-1`.
2. **하나라도 0이 아니고 화면(frontend)만 문제**: backend는 병합 SHA 이미지 그대로 다시 올리고(`$DC start xops-service`,
   누출 차단 유지) frontend만 이전 SHA로 다시 빌드한다(`$DC up -d --no-deps --build frontend` 후 `docker restart platform-frontend-1`).
   플래그를 0으로 내려도 만들어진 기록은 남지만 실데이터 모드·헤더 없는 호출에는 보이지 않는다.
3. **하나라도 0이 아니고 backend 자체 문제로 이전 SHA가 필요**, 또는 확인 결과가 불명확: 이전 SHA는 누출 차단이 없으므로
   backend만 되돌리지 않는다. 7.2의 `xops-data.tgz`로 볼륨을 백업 시점으로 복원(이력 제거)한 뒤 이전 SHA로 되돌린다.
   복원은 그 사이 실데이터 쓰기도 되돌리므로 별도 승인 후에만 한다. 승인 전에는 병합 SHA backend를 유지한다.
- 어느 경우든 복귀 뒤 헤더 없는 조회가 정상 응답이면서 누출 0인지 확인한다(HTTP 상태와 본문을 함께 본다):

```sh
curl -s -b jar -o cat.json -w '%{http_code}\n' "$B/api/v3/dataops/catalog?include_seed=true"   # 200
python3 -c "import json; rows=json.load(open('cat.json')); print('seed_or_mock', sum(1 for r in rows if r.get('is_seed')))"   # 0
TOKEN=$(curl -s -b jar -X POST "$B/api/v3/dataops/token" | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
curl -s -b jar -H "Authorization: Bearer $TOKEN" "$B/api/v3/dataops/apis" | python3 -c "import json,sys; print('mock_apis', sum(1 for r in json.load(sys.stdin) if r.get('mode')=='mock'))"   # 0
```
- 켠 뒤 화면에서 지운 이력은 다시 만들어지지 않는다(완료 표식). 다시 만들려면 볼륨 복원과 같은 수준의 결정이 필요하다.
