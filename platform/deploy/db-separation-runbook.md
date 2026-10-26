# DB Compose 분리 운영 절차 (project=population-db)

저장소 3종(pg·timescale·mongo)을 앱 project `platform`에서 떼어 project `population-db`로 관리한다.
데이터는 옮기지 않는다 — 같은 실볼륨을 같은 이미지의 새 컨테이너에 순서대로 다시 붙인다.

## 1. 구성

| 구분 | project | Compose 파일 | 서비스 |
|---|---|---|---|
| 앱 | `platform` | `compose.xops.yaml` + `compose.ec2.yaml` | xops-service, frontend, caddy, authelia |
| DB | `population-db` | `compose.db.yaml` + `compose.db.ec2.yaml` | pg, timescale, mongo |

- 공유망 `population-db-data`(bridge, 외부 소유): DB 3개는 이 망만, xops-service는 `platform_default` + 이 망.
  별칭 pg/timescale/mongo가 그대로라 `XOPS_*_DSN`은 바뀌지 않는다. frontend·Caddy·Authelia는 붙이지 않는다.
- 망은 `internal=false`로 만든다. Docker는 internal 망에만 붙은 컨테이너의 포트를 호스트에 공개하지 않으므로,
  서버 안 점검·적재용 `127.0.0.1:5433/5434/27017` 바인딩을 유지하려면 internal로 만들 수 없다.
  외부 노출은 지금과 같이 127.0.0.1 바인딩으로만 제한된다.
- 볼륨·이미지: `deploy/db.volumes.env`(예시 `deploy/db.volumes.env.example`)의 `POP_DB_*`. 볼륨은 external이라
  이름이 틀리거나 없으면 기동이 실패한다. 이미지는 RepoDigest + `pull_policy: never`로 고정한다.

명령 줄임(저장소 루트):

```bash
APP="docker compose -p platform --env-file platform/deploy/.env -f platform/compose.xops.yaml -f platform/compose.ec2.yaml"
DB="docker compose -p population-db --env-file platform/deploy/.env --env-file platform/deploy/db.volumes.env -f platform/compose.db.yaml -f platform/compose.db.ec2.yaml"
```

## 2. 금지

- 어떤 project든 `down`, `-v`, `--remove-orphans`, `prune`, 볼륨·이전 컨테이너 삭제.
- 같은 볼륨에 DB 프로세스 두 개(이전 컨테이너와 새 컨테이너 동시 실행).
- 분리와 동시에 DB 이미지 pull·업그레이드, init 재실행, 재적재(`scripts/load_*.py`), 마이그레이션.
- 앱만 바꿀 때 DB Compose를 읽거나 DB 컨테이너를 재시작하는 것. 앱은 `$APP up -d --no-deps <서비스>`만 쓴다.
- 분리 직후 `platform` project에는 이전 DB 컨테이너(platform-pg-1 등)가 정지 상태로 남아 orphan 경고가 뜬다.
  복귀 창 동안 보존해야 하므로 경고를 없애려고 `--remove-orphans`를 쓰지 않는다.

## 3. 최초 전환

1. 준비: 진행 중 학습·적재 작업 확인. 롤백 표시 `git tag -f pre-db-split-<날짜> HEAD`, 컨테이너·볼륨·망 목록 저장.
   `cp platform/deploy/db.volumes.env.example platform/deploy/db.volumes.env` 후 `docker inspect`로 값 대조.
2. 쓰기 중지: `docker stop platform-xops-service-1`.
3. 백업: `pg_dump -Fc`(pg·timescale), `mongodump --archive --gzip`, `platform_xops-data` tar. SHA256 기록,
   임시 컨테이너에 복구 시험 후 임시 자원만 삭제.
4. 망 단회 생성: `docker network create --driver bridge population-db-data`(이미 있으면 driver·internal 확인만).
5. 이전 DB 정지(삭제·이름 변경 금지): `docker stop platform-pg-1 platform-timescale-1 platform-mongo-1`,
   세 컨테이너 모두 `exited`인지 확인.
6. 새 DB 시작: `$DB config --quiet && $DB up -d`. 세 컨테이너 healthy, 이미지 ID·볼륨 name·경로가 이전과 같은지
   `docker inspect`로 대조. 로그에 init 실행(`/docker-entrypoint-initdb.d/` 실행 줄)이 보이면 실패로 보고 복귀.
7. 앱 재연결: `$APP up -d --no-deps xops-service` →
   `docker exec -i platform-xops-service-1 python - --min-pg-tables <전환 전 pg public 테이블 수> < platform/scripts/check_db_readiness.py`
   (모두 PASS, exit 0). 새 볼륨에 init이 다시 돌면 데모 테이블은 생기지만 적재 테이블(ext_*)은 없으므로
   최소 테이블 수로 걸러낸다. timescale·mongo는 3단계 건수 기준선과 대조해 확인한다. frontend nginx는 시작 때 한 번만 주소를 찾으므로 `docker restart platform-frontend-1`.
8. 검증: 실데이터 대표 소스 응답 `source_kind=database`(오류 이유 없음), 데모 소스 `source_kind=mock`,
   테이블·건수·컬렉션 기준선, xops.db 카탈로그·모델·작업 상태 동일. `GET /` 200만으로 통과시키지 않는다.

## 4. 복귀(문제 발생 시)

1. 쓰기 중지: `docker stop platform-xops-service-1`.
2. 새 DB 정지: `docker stop population-db-pg-1 population-db-timescale-1 population-db-mongo-1`, 모두 `exited` 확인.
3. 이전 DB 시작: `docker start platform-pg-1 platform-timescale-1 platform-mongo-1` → healthy·데이터 확인.
4. 체크아웃을 `pre-db-split-<날짜>`로 되돌리고 이전 Compose로 앱만 재생성:
   `git checkout --detach pre-db-split-<날짜>` → `$APP up -d --no-deps xops-service`(이전 Compose는 DB 서비스가 있어도
   `--no-deps`라 DB를 다시 만들지 않는다) → `docker restart platform-frontend-1`.
5. 같은 볼륨을 이어 쓰므로 데이터 복원은 하지 않는다. 전환 뒤 쓰기가 있었다면 그 데이터도 그대로 이어진다.
   저장소 손상처럼 데이터 복원이 필요하면 별도 복구 계획으로 다룬다(백업 덮어쓰기를 자동 복귀에 넣지 않는다).

## 5. 보존 종료

이전 DB 컨테이너 3개와 `platform_default` 망은 관찰 기간 동안 정지 상태로 둔다. 정리 시점은 별도 운영 판단이며
이 문서가 삭제를 승인하지 않는다.
