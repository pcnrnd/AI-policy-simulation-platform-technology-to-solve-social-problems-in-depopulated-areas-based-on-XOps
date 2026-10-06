# EC2 로그인 게이트 런북 (Caddy + Authelia)

플랫폼을 외부에 공개할 때 앞단에 로그인 화면을 붙여, 로그인한 계정만 화면과 `/api`에 닿게 한다.
앱 코드(`platform/backend`, `platform/frontend`)와 기존 `compose.xops.yaml`·`nginx.conf`·Dockerfile은 바꾸지 않고,
`platform/compose.ec2.yaml`을 겹쳐 쓴다.

```
브라우저 ──HTTPS──▶ Caddy :443 ──(forward_auth)──▶ Authelia :9091  ← 세션 없으면 로그인 포털로
                      │
                      └──▶ frontend(nginx) :80 ──/api──▶ xops-service :8000 ──▶ pg / timescale / mongo
```

| 파일 | 역할 |
|---|---|
| `platform/compose.ec2.yaml` | Caddy·Authelia 추가, 앱 포트 비공개, DB 포트 127.0.0.1 한정, 비밀값 주입 |
| `platform/deploy/caddy/Caddyfile` | HTTPS 종단, `/authelia` 포털 프록시, 나머지 전부 `forward_auth` 보호 |
| `platform/deploy/authelia/configuration.yml` | 파일 사용자(argon2id), 기본 deny + 도메인 전체 one_factor, IP 차단 규칙 |
| `platform/deploy/authelia/users.example.yml` | 계정 형식 예시 (실제 `users.yml`은 커밋 금지) |
| `platform/deploy/.env.example` | 필요한 환경변수 목록 (실제 `.env`는 커밋 금지) |

- 로그인 포털: `https://<SITE_DOMAIN>/authelia/`
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

   ```sh
   LT="docker compose --env-file platform/deploy/.env -p popdecline-logintest -f platform/compose.xops.yaml -f platform/compose.ec2.yaml"
   $LT up -d --build
   ```

3. 확인한다(`-k`는 자체 CA 때문).

   ```sh
   B=https://pop.localtest.me
   curl -sk -o /dev/null -D - $B/                          # 302, location: $B/authelia/?rd=...
   curl -sk -o /dev/null -D - $B/api/v3/realdata/health    # 302 또는 401, 데이터 없음
   # 로그인 → 세션 쿠키 저장
   curl -sk -c jar -H 'Content-Type: application/json' \
        -d '{"username":"<ID>","password":"<PW>","keepMeLoggedIn":false}' $B/authelia/api/firstfactor
   curl -sk -b jar -o /dev/null -w '%{http_code}\n' $B/                          # 200
   curl -sk -b jar $B/api/v3/realdata/health                                     # 200 + JSON
   # 틀린 비밀번호 5회 → 이후 올바른 비밀번호도 거부(차단)
   docker ps --format '{{.Names}}\t{{.Ports}}'             # 8000·8080 없음, DB는 127.0.0.1만
   ```

   브라우저로도 `https://pop.localtest.me/`에 접속하면 로그인 화면이 뜬다(자체 CA 경고는 무시).

4. 정리: 테스트 프로젝트만 지운다. 다른 컨테이너·볼륨·이미지는 건드리지 않는다.

   ```sh
   $LT down -v
   ```

## 5. 서버 기동

```sh
DC="docker compose --env-file platform/deploy/.env -f platform/compose.xops.yaml -f platform/compose.ec2.yaml"
$DC up -d --build      # 기동·갱신
$DC ps                 # 상태
$DC logs -f caddy authelia
$DC down               # 중지(볼륨 유지)
```

- 서버에서 `down -v`는 쓰지 않는다. DB, 로그인 이력·차단 목록(`authelia-data`), 인증서(`caddy-data`)가 함께 지워진다.
- 실데이터 적재 스크립트는 DB 비밀번호가 바뀌었으므로 DSN을 넘긴다.
  `python platform/scripts/load_real_data.py --pg-dsn postgresql://xops:<DB_PASSWORD>@127.0.0.1:5433/xops_dataops`

## 6. 알아둘 점

- **알림(notifier)은 filesystem이다.** 공식 문서상 이 방식은 테스트용이며 운영에서는 권장하지 않는다.
  메일 없이 운영하려고 택했고, 비밀번호 재설정·변경을 막아 두어 운영 흐름에서는 알림이 쓰이지 않는다.
  사용자 셀프 재설정이 필요해지면 SMTP notifier로 바꾼다.
- 세션은 Authelia 기본값(만료 1시간, 무활동 5분)을 쓴다. 로그인 화면의 "Remember me"를 고르면 더 길게 유지된다.
- 인증은 1단계(비밀번호)다. 2단계(TOTP 등)가 필요하면 `access_control`의 `one_factor`를 `two_factor`로 바꾸고 SMTP를 붙인다.
- Authelia 기동 로그의 `chown: /config...: Read-only file system`은 설정 폴더를 읽기 전용으로 마운트해서 나는 경고다. 동작에는 영향이 없다.
