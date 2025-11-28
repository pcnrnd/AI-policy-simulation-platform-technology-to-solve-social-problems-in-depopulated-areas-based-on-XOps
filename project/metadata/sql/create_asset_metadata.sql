-- 1. 확장 기능 활성화 (유지)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. 자산 메타데이터 테이블 생성
CREATE TABLE IF NOT EXISTS asset_metadata (
    id BIGSERIAL PRIMARY KEY,
    
    -- [MinIO 연동 정보] Raw 데이터 위치
    bucket_name VARCHAR(64) NOT NULL,        -- 예: 'raw-data'
    file_path VARCHAR(255) NOT NULL,         -- 예: 'images/2024/site_a_001.jpg'
    file_type VARCHAR(50) NOT NULL,          -- 예: 'image/jpeg', 'pdf', 'csv' (다양한 유형 대응)
    file_size BIGINT,                        -- 파일 크기 (bytes)

    -- [빠른 검색용 정보] 아직 벡터/좌표가 없을 때 키워드 검색용
    -- 예: {"site": "A구역", "manager": "김철수", "tags": ["균열", "누수"]}
    search_tags JSONB DEFAULT '{}',          

    -- [분석 결과 데이터] (초기엔 NULL, 추후 AI 분석 후 update)
    geom GEOMETRY(POINT, 4326),              -- 지도 좌표 (추후 입력)
    embedding VECTOR(512),                   -- 이미지 벡터 (추후 입력)
    
    -- [관리 정보]
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_status VARCHAR(20) DEFAULT 'PENDING' -- 상태 관리 (PENDING -> PROCESSED)
);

-- 3. 인덱스 설정
-- (1) 빠른 텍스트/태그 검색을 위한 JSONB 인덱스 (현재 가장 중요)
CREATE INDEX idx_asset_meta_tags ON asset_metadata USING GIN (search_tags);

-- (2) MinIO 경로 검색용 (중복 업로드 방지 등)
CREATE UNIQUE INDEX idx_asset_path ON asset_metadata (bucket_name, file_path);

-- (3) 추후 분석 완료 시 사용할 벡터/공간 인덱스 (데이터 들어오면 작동)
CREATE INDEX ON asset_metadata USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON asset_metadata USING gist (geom);