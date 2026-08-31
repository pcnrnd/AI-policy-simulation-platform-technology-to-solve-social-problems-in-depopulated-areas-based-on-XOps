-- DataOps 카탈로그 ds_05_smartfarm 의 TimescaleDB 대상 스키마.
-- 시계열 저장소로 분리한 이유: 카탈로그가 소스 유형으로 Adapter 를 나누므로
-- (adapters.dsn_for) '다기종 데이터 관리' 실증에는 별 인스턴스가 맞다.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ds_05_smartfarm
-- range 컬럼이 farm_id(문자열)라 하이퍼테이블 파티션 키로는 쓸 수 없다.
-- 시계열 축은 measured_at 을 별도로 두고, 조회 범위는 farm_id 로 건다.
CREATE TABLE IF NOT EXISTS ts_smartfarm_yield (
    measured_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    farm_id            VARCHAR(12)      NOT NULL,
    region_code        VARCHAR(5)       NOT NULL,
    crop_type          VARCHAR(20)      NOT NULL,
    cultivation_area   DOUBLE PRECISION NOT NULL,
    investment_amount  BIGINT           NOT NULL,
    yield_amount       DOUBLE PRECISION NOT NULL,
    employment_count   INTEGER          NOT NULL
);

-- 하이퍼테이블 전환 (이미 전환됐으면 무시)
SELECT create_hypertable('ts_smartfarm_yield', 'measured_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ix_smartfarm_farm_id ON ts_smartfarm_yield (farm_id, measured_at DESC);

-- 최소 시드 — farm_id 는 range(NW-SF-001~NW-SF-128) 안에 들어와야 한다.
INSERT INTO ts_smartfarm_yield (
    measured_at, farm_id, region_code, crop_type, cultivation_area,
    investment_amount, yield_amount, employment_count
)
SELECT
    now() - (n || ' days')::interval,
    'NW-SF-' || lpad((n % 128 + 1)::text, 3, '0'),
    '45190',
    CASE WHEN n % 3 = 0 THEN 'tomato' WHEN n % 3 = 1 THEN 'strawberry' ELSE 'paprika' END,
    1200.0 + n * 10,
    80000000 + n * 1000000,
    3200.5 + n * 12.5,
    4 + (n % 7)
FROM generate_series(0, 59) AS n
WHERE NOT EXISTS (SELECT 1 FROM ts_smartfarm_yield);
