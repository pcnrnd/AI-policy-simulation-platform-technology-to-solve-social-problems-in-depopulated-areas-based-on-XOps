INSERT INTO asset_metadata (
    id, bucket_name, file_path, file_type, file_size, search_tags, processed_status
) VALUES 
-- 1. 이미지 데이터 (AI 분석 예정 자산 - 기존 일반)
(
    12, 
    'raw-data', 
    'images/siteA_crack_001.jpg', 
    'image/jpeg', 
    5500000, 
    '{"asset_type": "inspection_photo", "site": "A구역", "issue_tags": ["균열", "심각", "긴급"], "department": "시설관리팀"}'::jsonb,
    'PENDING'
),
(
    13, 
    'raw-data', 
    'images/drone_view_B.png', 
    'image/png', 
    12000000, 
    '{"asset_type": "aerial_photo", "site": "B구역", "view_type": "광역", "equipment": "드론", "department": "기획조사부"}'::jsonb,
    'PENDING'
),

-- 2. 센서/로그 데이터 (기존 일반)
(
    14, 
    'log-data', 
    'logs/temp_sensor_202511.csv', 
    'text/csv', 
    800000, 
    '{"asset_type": "sensor_log", "sensor_id": "T-005", "data_metric": "온도", "period": "202511"}'::jsonb,
    'PENDING'
),
(
    15, 
    'log-data', 
    'logs/vibr_alert_1003.json', 
    'application/json', 
    50000, 
    '{"asset_type": "alert_log", "sensor_id": "V-101", "data_metric": "진동", "trigger_date": "2025-10-03", "status": "critical"}'::jsonb,
    'PENDING'
),

-- 3. GIS/CAD 도면 데이터 (기존 일반)
(
    16, 
    'gis-data', 
    'maps/siteA_cad_v3.dxf', 
    'application/dxf', 
    1500000, 
    '{"asset_type": "CAD_drawing", "site": "A구역", "version": "v3.0", "department": "설계팀"}'::jsonb,
    'PENDING'
),
(
    17, 
    'gis-data', 
    'maps/soil_map_v1.shp', 
    'application/octet-stream', 
    2300000, 
    '{"asset_type": "GIS_layer", "map_topic": "지반조사", "coverage": "전체", "data_format": "Shapefile"}'::jsonb,
    'PENDING'
),

-- 4. 인구감소/소외 지역 특징 반영 이미지 데이터 (사람 이름 제외)
(
    18, 
    'raw-data', 
    'images/rural_infra_crack.jpg', 
    'image/jpeg', 
    6200000, 
    '{"asset_type": "inspection_photo", "site": "D지역_노후시설", "issue_tags": ["균열", "파손", "긴급"], "department": "시설관리팀", "location_tags": ["인구감소지역", "노후화지구"]}'::jsonb,
    'PENDING'
),
(
    19, 
    'raw-data', 
    'images/remote_site_view.png', 
    'image/png', 
    11500000, 
    '{"asset_type": "aerial_photo", "site": "E지역_외곽", "view_type": "광역", "equipment": "드론", "department": "기획조사부", "location_tags": ["교통소외지역", "접근성_낮음"]}'::jsonb,
    'PENDING'
),

-- 5. 인구감소/소외 지역 특징 반영 문서/로그 데이터 (사람 이름 제외)
(
    20, 
    'doc-data', 
    'documents/area_dev_plan_2026.pdf', 
    'application/pdf', 
    3800000, 
    '{"asset_type": "official_document", "document_type": "개발계획", "target_area": "E지역", "department": "도시계획과", "location_tags": ["소외지역", "균형발전"]}'::jsonb,
    'PENDING'
),
(
    21, 
    'data-archive', 
    'documents/pop_stat_2025.csv', 
    'text/csv', 
    950000, 
    '{"asset_type": "statistic_data", "data_topic": "인구통계", "data_year": 2025, "department": "통계청_협업", "location_tags": ["인구감소지역", "고령화"]}'::jsonb,
    'PENDING'
),
(
    22, 
    'gis-data', 
    'maps/social_infra_v2.shp', 
    'application/octet-stream', 
    4100000, 
    '{"asset_type": "GIS_layer", "map_topic": "사회기반시설", "coverage": "F시", "department": "GIS분석팀", "location_tags": ["인구감소지역", "생활SOC"]}'::jsonb,
    'PENDING'
),

-- 6. 한국 지역 특성 반영 데이터 (행정 구역 명시)
(
    23, 
    'gis-data', 
    'docs/seoul_infra_map.pdf', 
    'application/pdf', 
    5100000, 
    '{"asset_type": "map_data", "province": "서울특별시", "city": "강남구", "department": "도시계획과", "location_tags": ["수도권", "고밀도지역"]}'::jsonb,
    'PENDING'
),
(
    24, 
    'raw-data', 
    'images/busan_port_photo.jpg', 
    'image/jpeg', 
    7800000, 
    '{"asset_type": "inspection_photo", "province": "부산광역시", "city": "해운대구", "issue_tags": ["시설물_양호", "정기점검"], "department": "항만관리부", "location_tags": ["대도시권", "항만지역"]}'::jsonb,
    'PENDING'
),
(
    25, 
    'doc-data', 
    'reports/rural_pop_outflow.pdf', 
    'application/pdf', 
    4300000, 
    '{"asset_type": "analysis_report", "province": "전라남도", "city": "A군", "document_topic": "인구유출", "department": "지역정책팀", "location_tags": ["인구감소지역", "지방소멸위험"]}'::jsonb,
    'PENDING'
),
(
    26, 
    'log-data', 
    'logs/facility_aging_temp.csv', 
    'text/csv', 
    990000, 
    '{"asset_type": "sensor_log", "province": "경상북도", "city": "B시", "data_metric": "온도", "sensor_id": "T-100", "location_tags": ["인구감소지역", "노후화시설"]}'::jsonb,
    'PENDING'
),
(
    27, 
    'raw-data', 
    'images/remote_access_road.jpg', 
    'image/jpeg', 
    7500000, 
    '{"asset_type": "inspection_photo", "province": "강원도", "city": "C군", "issue_tags": ["도로_파손", "접근성_문제"], "department": "도로관리부", "location_tags": ["교통소외지역", "산간지역"]}'::jsonb,
    'PENDING'
),
(
    28, 
    'gis-data', 
    'maps/cultural_gaps.shp', 
    'application/octet-stream', 
    3600000, 
    '{"asset_type": "GIS_layer", "map_topic": "문화시설", "province": "충청남도", "city": "D시", "department": "정책분석팀", "location_tags": ["문화소외지역", "생활SOC_취약"]}'::jsonb,
    'PENDING'
);