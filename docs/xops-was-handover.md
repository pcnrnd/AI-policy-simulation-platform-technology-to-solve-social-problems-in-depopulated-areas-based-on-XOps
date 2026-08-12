# XOps 서비스 WAS 구성 인수인계

> 작성일: 2026-08-12 · 대상: `platform/` (xops-service + frontend)

## 요약

xops-service 스택은 이미 WEB–WAS–DB 3-tier 구조로 구성되어 있다.
Python 스택에서는 별도 WAS 제품(Tomcat/JEUS 등)을 얹는 것이 아니라 **ASGI 서버(uvicorn)가 WAS 역할**을 담당한다.

## 계층 구성

| 계층 | 구성 요소 | 파일 | 핵심 내용 |
|------|-----------|------|-----------|
| WEB 서버 | nginx (frontend 컨테이너, :8080) | `platform/frontend/nginx.conf` | 정적 파일 서빙 + `/api` → `http://xops-service:8000` 리버스 프록시 |
| WEB 서버 | nginx 이미지 빌드 | `platform/frontend/Dockerfile` | Vite 빌드(dist) + nginx.conf 탑재 (멀티스테이지) |
| **WAS** | uvicorn + FastAPI (:8000) | `platform/backend/xops-service/Dockerfile` | `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]` |
| DB | SQLite (`xops-data` 볼륨) | `platform/compose.xops.yaml` | 사용자 소스·모델 버전·실행 이력 영속화 |

## 요청 흐름

```
브라우저 → nginx(:8080)
            ├─ 정적 자산(SPA): 직접 서빙 (try_files → index.html 폴백)
            └─ /api/* : compose 내부 네트워크로 uvicorn(:8000) 프록시
                          → FastAPI(xops-service) → SQLite(/app/data)
```

- 동일 오리진으로 처리되므로 CORS 설정 불필요.
- xops-service의 `8000:8000` 포트 노출은 디버깅용 직접 접근 경로 (프론트는 내부 네트워크 사용).

## 기동 방법

```powershell
docker compose -f platform/compose.xops.yaml up -d --build
# 접속: http://localhost:8080 (프론트) / http://localhost:8000/docs (백엔드 직접)
```

## 프로덕션 전환 시 남은 작업 (미적용)

1. **멀티 프로세스 워커** — 현재 uvicorn 단일 프로세스. WAS Dockerfile CMD를 다음으로 교체:
   ```dockerfile
   CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000"]
   ```
   (`requirements.txt`에 `gunicorn` 추가 필요)
2. **WEB–WAS 분리 완성** — `compose.xops.yaml`에서 xops-service의 `ports: 8000:8000` 제거 → 모든 트래픽이 nginx 경유.
3. **환경 변수** — prod 전환 시 `XOPS_ENVIRONMENT=prod` + `XOPS_JWT_SECRET`(필수) + `XOPS_CLIENT_ID/SECRET` 설정 (compose 주석 참조).
