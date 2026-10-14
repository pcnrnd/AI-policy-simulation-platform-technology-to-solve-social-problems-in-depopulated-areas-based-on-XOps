# 파트너 모듈 연결 대조표

2026-10-14 · Phase D 준비 계약. **현재 실제 파트너 연동은 미완료**다. dts의 integration-test profile은 저장소의 라우팅 검증용 더미 2종이며 실제 부서 FE/BE나 실데이터 결과가 아니다.

| 확인 항목 | 우리 준비 상태 | 사용자/파트너가 제공·확인할 정보 | 실제 통합 판정 |
|---|---|---|---|
| 도메인·DNS·TLS | SITE_DOMAIN env, 기존 Caddy HTTPS 유지 | 보유 도메인·공개 URL·DNS/인증서 확인(사용자 담당) | 대기 |
| 로그인 public URL | AUTH_PUBLIC_URL, 같은 origin /authelia 기본·기존 env 호환 | 실제 portal public URL, 쿠키 domain, 로그인 복귀 정책 | 대기 |
| dashboard FE | public path·origin/port·upstream base path env, 미설정 안내 | 실제 FE origin·port·base path·자산 경로·deep link | 대기 |
| dashboard API | 전용 prefix·origin/port·upstream base env | 실제 API origin·port·prefix·메서드·요청/응답 스키마 | 대기 |
| reporting FE/API | dashboard와 같은 분리 구조 | 실제 FE/BE 값, 다운로드 형식·파일명·권한 | 대기 |
| 인증 identity | Caddy가 입력 Remote-* 제거 후 Authelia 값만 전달 | 신뢰 가능한 proxy 네트워크·그룹/권한 매핑 | 대기 |
| module 자체 인증 | Cookie/Authorization 미전달 | 별도 BFF/SSO·서버 credential 방식; 브라우저 secret 금지 | 대기 |
| 모드 | mock/real 명시, 지원 미확인 안내·real fallback0 | 각 모드 구현과 X-Data-Mode 우선 계약, mode query 일치 | 대기 |
| 쿼리·인코딩·새로고침 | 공개 prefix 치환·raw URI/query 보존, 중복/모호 mode 거절 | region·기간·필터 query, percent encoding, 직접 접속·SPA fallback | 대기 |
| 쿠키·응답 | upstream Cookie 미전달, Set-Cookie/Clear-Site-Data/X-Accel/Refresh 차단 | module 쿠키 필요 여부와 별도 승인된 범위, redirect·cache 계약 | 대기 |
| JS 저장소 격리 | dummy 앱 저장소 접근0, namespace 계약 | 같은 origin은 JS storage 강제 격리 불가; 신뢰 검토·별도 origin/격리 설계 | **실통합 보류** |
| 서버 간 readonly BFF | 기존 xops readonly token 계약 유지 | 파트너 서버에 credential 보관·scope·갱신·오류 처리 | 대기 |
| HTTPS·보안 헤더 | upstream HTTPS CA 검증, no-store/nosniff/frame 정책 | 실제 인증서 체인·CSP·WebSocket·CORS 필요 여부 | 대기 |
| endpoint 검수 | 더미에서만 전송·라우팅 증거 확보 예정 | 실제 주소별 Auth·mode·asset·API·다운로드 검수 | 대기 |

모든 행의 정보가 확인되기 전 더미 주소를 실제 주소로 간주하지 않는다. 실제 partner JS는 저장소 격리 설계 확인 전 같은 origin에서 실행하지 않는다. dev 개방 token 발급은 readonly 보장을 제공하지 않으므로 실제 읽기 연동은 기존 BFF 계약을 따라야 한다.

배포 절차·env 표·기본 복귀 명령은 [외부 모듈 런북](deploy/external-modules.md), API 계약은 [부서 연동 문서](xops-department-integration.md)를 사용한다.
