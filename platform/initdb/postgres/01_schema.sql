-- DataOps 카탈로그(ds_01~ds_04, ds_06)의 PostgreSQL·PostGIS 대상 스키마.
-- mock_data.json metadata_schemas 의 object/columns/range 와 1:1로 맞춘다.
-- postgis 이미지를 쓰므로 PostgreSQL 과 PostGIS 를 한 인스턴스에서 함께 제공한다.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ds_01_resident_registry
CREATE TABLE IF NOT EXISTS tb_resident_movement (
    reg_date        VARCHAR(8)  NOT NULL,
    region_code     VARCHAR(5)  NOT NULL,
    in_flow_count   INTEGER     NOT NULL,
    out_flow_count  INTEGER     NOT NULL,
    age_group       VARCHAR(3)  NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_resident_movement_reg_date ON tb_resident_movement (reg_date);

-- ds_02_local_welfare
CREATE TABLE IF NOT EXISTS tb_welfare_budget (
    year                     VARCHAR(4) NOT NULL,
    region_code              VARCHAR(5) NOT NULL,
    welfare_budget           BIGINT     NOT NULL,
    infant_support_budget    BIGINT     NOT NULL,
    youth_employment_budget  BIGINT     NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_welfare_budget_year ON tb_welfare_budget (year);

-- ds_03_industrial_factories
CREATE TABLE IF NOT EXISTS tb_factory_registry (
    factory_id      VARCHAR(12) NOT NULL,
    region_code     VARCHAR(5)  NOT NULL,
    employee_count  INTEGER     NOT NULL,
    industry_type   VARCHAR(6)  NOT NULL,
    annual_revenue  BIGINT      NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_factory_registry_factory_id ON tb_factory_registry (factory_id);

-- ds_04_spatial_geojson (PostGIS)
CREATE TABLE IF NOT EXISTS geo_grid_cells (
    grid_id             VARCHAR(10) NOT NULL,
    region_code         VARCHAR(5)  NOT NULL,
    geom_polygon        GEOMETRY(Polygon, 4326),
    population_density  DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_geo_grid_cells_grid_id ON geo_grid_cells (grid_id);
CREATE INDEX IF NOT EXISTS ix_geo_grid_cells_geom ON geo_grid_cells USING GIST (geom_polygon);

-- ds_06_settlement_facility
CREATE TABLE IF NOT EXISTS tb_settlement_facility (
    facility_id       VARCHAR(12) NOT NULL,
    region_code       VARCHAR(5)  NOT NULL,
    facility_type     VARCHAR(10) NOT NULL,
    rental_units      INTEGER     NOT NULL,
    operating_budget  BIGINT      NOT NULL,
    occupancy_rate    DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_settlement_facility_id ON tb_settlement_facility (facility_id);

-- 최소 시드 — 카탈로그 조회가 빈 결과가 아니어야 연동을 확인할 수 있다.
-- reg_date 는 range(20210101~20261231) 안에 들어와야 WHERE BETWEEN 에 걸린다.
INSERT INTO tb_resident_movement (reg_date, region_code, in_flow_count, out_flow_count, age_group)
SELECT
    to_char(DATE '2021-01-01' + (n * 30), 'YYYYMMDD'),
    CASE WHEN n % 2 = 0 THEN '45190' ELSE '46900' END,
    120 + n,
    180 + n,
    CASE WHEN n % 3 = 0 THEN '20s' WHEN n % 3 = 1 THEN '40s' ELSE '60s' END
FROM generate_series(0, 59) AS n
WHERE NOT EXISTS (SELECT 1 FROM tb_resident_movement);

INSERT INTO tb_welfare_budget (year, region_code, welfare_budget, infant_support_budget, youth_employment_budget)
SELECT (2017 + n)::text, '45190', 100000000 + n * 5000000, 20000000 + n * 1000000, 15000000 + n * 800000
FROM generate_series(0, 9) AS n
WHERE NOT EXISTS (SELECT 1 FROM tb_welfare_budget);

INSERT INTO tb_factory_registry (factory_id, region_code, employee_count, industry_type, annual_revenue)
SELECT 'F-' || lpad((102 + n)::text, 6, '0'), '45190', 10 + n, 'C' || (10 + n % 20)::text, 500000000 + n * 10000000
FROM generate_series(0, 39) AS n
WHERE NOT EXISTS (SELECT 1 FROM tb_factory_registry);

INSERT INTO geo_grid_cells (grid_id, region_code, geom_polygon, population_density)
SELECT
    'GRID-' || lpad((n + 1)::text, 5, '0'),
    '45190',
    ST_SetSRID(ST_MakeEnvelope(127.0 + n * 0.01, 35.0 + n * 0.01, 127.01 + n * 0.01, 35.01 + n * 0.01), 4326),
    45.5 + n
FROM generate_series(0, 39) AS n
WHERE NOT EXISTS (SELECT 1 FROM geo_grid_cells);

INSERT INTO tb_settlement_facility (facility_id, region_code, facility_type, rental_units, operating_budget, occupancy_rate)
SELECT 'SA-FC-' || lpad((n + 1)::text, 4, '0'), '46900', '임대주택', 100 + n, 15000000 + n * 500000, 0.80 + (n % 20) * 0.01
FROM generate_series(0, 39) AS n
WHERE NOT EXISTS (SELECT 1 FROM tb_settlement_facility);
