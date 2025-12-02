WITH vector_series AS (
    -- 512개의 0.0001 간격의 숫자를 가진 기본 벡터를 생성
    SELECT array_agg((i * 0.0001)::double precision ORDER BY i) AS base_v
    FROM generate_series(0, 511) AS i
),
transformed_vectors AS (
    -- CSV (유사 벡터)
    SELECT 1 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.01 AS offset, 1.0 AS scale UNION ALL
    SELECT 2 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.02 AS offset, 1.0 AS scale UNION ALL
    SELECT 3 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.03 AS offset, 1.0 AS scale UNION ALL
    SELECT 4 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.04 AS offset, 1.0 AS scale UNION ALL
    SELECT 5 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.05 AS offset, 1.0 AS scale UNION ALL
    SELECT 6 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.06 AS offset, 1.0 AS scale UNION ALL
    -- PDF (유사 벡터)
    SELECT 7 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.5 AS offset, 0.5 AS scale UNION ALL
    SELECT 8 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.55 AS offset, 0.5 AS scale UNION ALL
    SELECT 9 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.6 AS offset, 0.5 AS scale UNION ALL
    SELECT 10 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.65 AS offset, 0.5 AS scale UNION ALL
    SELECT 11 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.7 AS offset, 0.5 AS scale UNION ALL
    -- IMAGE (독립 벡터)
    SELECT 12 AS id, (SELECT base_v FROM vector_series) AS v_array, 0.8 AS offset, 0.3 AS scale
)
UPDATE asset_metadata a
SET embedding = (
    -- 기본 벡터에 ID별 offset과 scale을 적용하여 512차원 벡터 생성
    SELECT ARRAY(
        SELECT v_element * t.scale + t.offset
        FROM unnest(t.v_array) AS v_element
    )
)::VECTOR(512)
FROM transformed_vectors t
WHERE a.id = t.id;