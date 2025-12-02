WITH query_vector AS (
    -- ID 12번의 벡터를 쿼리 벡터로 정의
    SELECT embedding AS target_vector
    FROM asset_metadata
    WHERE id = 12
)
SELECT
    a.id,	
    a.file_path,
    a.search_tags ->> 'company' AS company,
    a.search_tags ->> 'asset_type' AS asset_type,
    a.embedding <=> q.target_vector AS cosine_distance
FROM
    asset_metadata a, query_vector q
WHERE
    a.search_tags ? 'document_type' 
ORDER BY
    cosine_distance 
LIMIT 5;