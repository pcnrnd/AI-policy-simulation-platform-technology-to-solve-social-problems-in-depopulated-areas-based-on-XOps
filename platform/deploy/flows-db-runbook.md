# flows DB 별도 Compose 운영 절차 (project=population-flows)

`flows` 스키마(63개 테이블)를 담는 PostgreSQL을 기존 DB(`population-db`)·앱(`platform`)과 분리한 project로 운영한다.
이 문서는 DB 기동과 데이터 이관까지만 다룬다. 앱 코드 연동은 포함하지 않는다.

## 1. 구성

| 항목 | 값 |
|---|---|
| project | `population-flows` (파일: `compose.flows.yaml` + 운영 `compose.flows.ec2.yaml`) |
| 서비스·이미지 | `flows-db`, `postgis/postgis:16-3.4` (기존 `pg`와 같은 이미지; 운영은 `POP_DB_PG_IMAGE` RepoDigest, `pull_policy: never`) |
| 데이터 볼륨 | 외부 볼륨 `POP_DB_FLOWS_VOLUME` (`deploy/db.volumes.env`) — 없으면 기동 실패 |
| 망 | 외부 망 `population-db-data`, 별칭 `flows-db` |
| 호스트 포트 | 기본 없음. 운영 오버레이만 `127.0.0.1:5435` (5433·5434·27017과 겹치지 않음) |
| 메모리 | `shared_buffers=64MB`, `max_connections=20` (EC2 총 1.9GB·가용 약 1.1GB에서 기존 DB와 함께 운용) |

명령 줄임(저장소 루트, 운영):

```bash
FDB="docker compose -p population-flows --env-file platform/deploy/.env --env-file platform/deploy/db.volumes.env -f platform/compose.flows.yaml -f platform/compose.flows.ec2.yaml"
```

## 2. 금지

- `-p population-db`로 이 파일 실행(기존 DB project와 섞임).
- `down -v`, `--remove-orphans`, `prune`, 볼륨 삭제.
- 적재가 끝난 볼륨에 대한 재적재·`DROP`(덮어쓰기 전에 별도 백업).

## 3. 최초 기동

1. 사전 확인: 가용 메모리, `docker network inspect population-db-data`(없으면 `db-separation-runbook.md`의 망 생성 단계로 한 번만 생성).
2. 볼륨 생성: `docker volume create <POP_DB_FLOWS_VOLUME 값>`. `deploy/db.volumes.env`에 볼륨 이름을 채우고, `POP_DB_PG_IMAGE`는 서버의 기존 `pg` 컨테이너 RepoDigest 그대로 둔다(서버에 이미지가 이미 있어야 함).
3. 검증과 기동: `$FDB config --quiet && $FDB up -d` → `docker ps`에서 `population-flows-flows-db-1`이 healthy가 될 때까지 대기.

## 4. 데이터 이관 (MariaDB → PostgreSQL)

원본 MariaDB가 있는 PC에서 변환한 뒤 덤프만 서버로 옮긴다.

1. PC에서 임시 PostgreSQL을 띄우고 `pgloader`로 MariaDB `flows`를 변환 적재한다.
2. `pg_dump -Fc -U xops flows > flows.dump`로 덤프한다.
3. 덤프 파일을 서버로 복사한다.
4. 서버에서 `docker exec -i population-flows-flows-db-1 pg_restore -U xops -d flows --no-owner < flows.dump`로 복원한다. 비밀번호는 명령줄·로그에 남기지 않는다.
5. 대조: `information_schema.tables`에서 해당 스키마 테이블 수가 63인지, 테이블별 행 수가 원본과 같은지 확인한다.

주의: 이 이관 절차는 아직 실제 데이터로 시험하지 않았다. 처음 이관할 때 63개 테이블 수와 테이블별 행 수를 원본과 대조하고, pgloader가 타입 경고를 낸 테이블은 복원 뒤 컬럼 타입과 값을 따로 확인한다.

## 5. 중지·되돌리기

`$FDB stop`. 데이터는 볼륨에 남으므로 `$FDB start`로 이어진다. 볼륨은 지우지 않는다.

## 6. 앱 연동 시 접속 정보

| 항목 | 값 |
|---|---|
| 망 | `population-db-data` (앱 컨테이너를 이 망에 붙인다) |
| 호스트 | `flows-db` |
| 포트 | `5432` |
| DB | `flows` |
| 스키마 | 변환 적재 결과에 따른다(이관 후 `\dn`으로 확인) |
| 사용자 | `xops` (비밀번호는 `DB_PASSWORD`; 전용 계정 분리는 연동 때 결정) |

서버 안 점검은 `127.0.0.1:5435`로 한다(외부 공개 없음). 비밀번호는 볼륨을 처음 만들 때만 적용된다.
