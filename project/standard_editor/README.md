# DataOps Standard Editor

데이터옵스를 위한 표준 명세 편집기/해석기

## 프로젝트 개요

고품질 데이터 생산을 위해 Data Repository, Checkout, Commit, Update, Add, Branch, Merge 협업 기능을 제공하는 웹 애플리케이션입니다.

## 주요 기능

- **Repository**: 데이터 저장소 초기화 및 관리
- **Checkout**: Repository에서 로컬로 프로젝트 복사
- **Commit**: 로컬의 코드를 Repository에 저장
- **Update**: 로컬에서 작업중인 코드를 Repository로 저장
- **Add**: 로컬에서 새로운 파일을 추가했을 때 Repository에 등록
- **Branch**: Root 프로젝트로부터 파생된 프로젝트 생성 및 버전 관리
- **Merge**: Branch에서 진행하던 작업을 Root 프로젝트와 합침

## 기술 스택

### Backend
- FastAPI (Python)
- DVC (Data Version Control)
- Git

### Frontend
- React 18
- Vite
- React Router
- Axios
- Lucide React (아이콘)

## 프로젝트 구조

```
project/standard_editor/
├── backend/
│   ├── api/              # API 엔드포인트
│   ├── config/           # 설정 관리
│   ├── core/             # 핵심 모듈 (예외, 로거)
│   ├── models/           # Pydantic 모델
│   ├── repositories/     # 명령어 실행 추상화
│   ├── services/         # 비즈니스 로직
│   └── main.py           # FastAPI 앱 진입점
└── frontend/
    ├── src/
    │   ├── components/   # 재사용 가능한 컴포넌트
    │   ├── pages/        # 페이지 컴포넌트
    │   ├── services/     # API 통신
    │   └── styles/       # 스타일 파일
    └── package.json
```

## 설치 및 실행

### Backend

1. 의존성 설치:
```bash
cd project/standard_editor/backend
pip install -r requirements.txt
```

**중요**: `pydantic-settings` 패키지가 설치되어 있는지 확인하세요:
```bash
pip install pydantic-settings
```

2. 서버 실행 (권장 방법):
```bash
# backend 디렉토리에서 실행
cd project/standard_editor/backend
python main.py
```

또는 별도 실행 스크립트 사용:
```bash
cd project/standard_editor/backend
python run.py
```

또는 uvicorn 직접 실행:
```bash
cd project/standard_editor/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**주의**: 반드시 `backend` 디렉토리에서 실행해야 합니다. 상대 import 경로 때문입니다.

백엔드 서버는 `http://localhost:8000`에서 실행됩니다.
API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

### Frontend

1. 의존성 설치:
```bash
cd project/standard_editor/frontend
npm install
```

2. 개발 서버 실행:
```bash
npm run dev
```

프론트엔드는 `http://localhost:5173`에서 실행됩니다.

## 사용 방법

### 1. Repository 초기화

1. Repository 페이지로 이동
2. 저장소 경로 입력 (예: `/path/to/repository`)
3. 원격 저장소 URL 입력 (선택, 예: `s3://bucket-name`)
4. "초기화" 버튼 클릭

### 2. 파일 추가

1. Commit 페이지로 이동
2. 저장소 경로 입력
3. 추가할 파일 경로 입력 (쉼표로 구분)
4. "파일 추가" 버튼 클릭

### 3. 변경사항 커밋

1. Commit 페이지로 이동
2. 저장소 경로 입력
3. 커밋 메시지 입력
4. 필요시 "원격 저장소로 푸시" 체크
5. "커밋" 버튼 클릭

### 4. 브랜치 관리

1. Branch 페이지로 이동
2. 저장소 경로 입력
3. 브랜치 이름 입력
4. "새 브랜치 생성" 체크 (새 브랜치 생성 시)
5. "체크아웃" 버튼 클릭

### 5. 브랜치 병합

1. Branch 페이지로 이동
2. 저장소 경로 입력
3. 소스 브랜치와 타겟 브랜치 입력
4. "병합" 버튼 클릭

## API 엔드포인트

### Repository
- `POST /api/v1/repository/init` - Repository 초기화
- `POST /api/v1/repository/checkout` - 브랜치 체크아웃
- `POST /api/v1/repository/commit` - 변경사항 커밋
- `POST /api/v1/repository/add` - 파일 추가
- `POST /api/v1/repository/update` - 원격 저장소에서 업데이트
- `POST /api/v1/repository/merge` - 브랜치 병합
- `GET /api/v1/repository/branches` - 브랜치 목록 조회
- `GET /api/v1/repository/status` - 저장소 상태 조회

## 환경 변수

백엔드 설정은 `backend/config/settings.py`에서 관리됩니다.
필요시 `.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다.

## 주의사항

- DVC와 Git이 시스템에 설치되어 있어야 합니다.
- 원격 저장소 사용 시 MinIO 또는 S3 호환 저장소가 필요합니다.
- 저장소 경로는 절대 경로를 사용하는 것을 권장합니다.

## 개발 가이드

### 객체지향 설계 원칙

- **CommandExecutor**: 명령어 실행 추상화 계층
- **DVCService/GitService**: 각각의 도메인 서비스
- **RepositoryService**: 비즈니스 로직 통합 서비스
- **API Layer**: FastAPI 라우터를 통한 엔드포인트 제공

### 프론트엔드 구조

- **Services**: API 통신 로직
- **Components**: 재사용 가능한 UI 컴포넌트
- **Pages**: 페이지 레벨 컴포넌트

## 라이선스

ISC

