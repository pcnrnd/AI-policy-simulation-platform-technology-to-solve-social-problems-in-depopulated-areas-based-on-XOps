from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_search_tags',
    description='파일명 기반 search_tags 및 embedding 생성',
    start_date=pendulum.datetime(2025, 11, 27),
    schedule="@daily",
    tags=['metadata', 'tagging', 'embedding'],
    default_args=default_args
) as dag:

    @task.virtualenv(
        task_id="generate_search_tags",
        requirements=['psycopg2-binary']
    )
    def generate_search_tags():
        """간단한 태그 생성"""
        import psycopg2
        import json
        import re
        from pathlib import Path
        
        conn = psycopg2.connect(
            host='db', port=5432,
            database='population_metadata',
            user='postgres', password='postgres'
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, file_path, file_type
            FROM asset_metadata 
            WHERE search_tags = '{}'::jsonb
        """)
        
        for record_id, file_path, file_type in cursor.fetchall():
            tags = {}
            filename = Path(file_path).stem.lower()
            
            # 기본 태그
            if 'image' in file_type:
                tags['asset_type'] = 'image'
            elif 'csv' in file_type:
                tags['asset_type'] = 'timeseries_data'
            elif 'pdf' in file_type:
                tags['asset_type'] = 'document'
            
            # 키워드만 간단히 매칭
            if 'restaurant' in filename:
                tags['site'] = '음식점'
            if 'crack' in filename:
                tags['issue_tags'] = ['균열']
            if 'drone' in filename:
                tags['equipment'] = '드론'
            if 'sensor' in filename:
                tags['asset_type'] = 'sensor_log'
            
            # 연도
            year = re.search(r'(\d{4})', file_path)
            if year and 2000 <= int(year.group(1)) <= 2100:
                tags['data_year'] = year.group(1)
            
            if tags:
                cursor.execute("""
                    UPDATE asset_metadata 
                    SET search_tags = %s::jsonb
                    WHERE id = %s
                """, (json.dumps(tags, ensure_ascii=False), record_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True

    @task.virtualenv(
        task_id="generate_filename_embeddings",
        requirements=['sentence-transformers', 'psycopg2-binary', 'numpy']
    )
    def generate_filename_embeddings():
        """파일명 기반 embedding 생성"""
        from sentence_transformers import SentenceTransformer
        import psycopg2
        import numpy as np
        from pathlib import Path
        import re
        
        # 모델 로드
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        conn = psycopg2.connect(
            host='db', port=5432,
            database='population_metadata',
            user='postgres', password='postgres'
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, file_path 
            FROM asset_metadata 
            WHERE embedding IS NULL
            AND processed_status = 'PENDING'
            LIMIT 1000
        """)
        
        records = cursor.fetchall()
        
        if not records:
            print("처리할 레코드가 없습니다.")
            return 0
        
        def normalize_filename(file_path):
            """파일명 정규화"""
            filename = Path(file_path).stem
            text = filename.replace('_', ' ').replace('-', ' ')
            text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
            text = ' '.join(text.split())
            return text.lower()
        
        # 배치 처리
        texts = [normalize_filename(file_path) for _, file_path in records]
        embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)
        
        updated_count = 0
        for idx, (record_id, file_path) in enumerate(records):
            try:
                embedding = embeddings[idx]
                
                # 512차원 변환
                if len(embedding) > 512:
                    embedding = embedding[:512]
                elif len(embedding) < 512:
                    padding = np.zeros(512 - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                
                embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                
                cursor.execute("""
                    UPDATE asset_metadata 
                    SET embedding = %s::vector(512),
                        processed_status = 'PROCESSED'
                    WHERE id = %s
                """, (embedding_str, record_id))
                updated_count += 1
                
            except Exception as e:
                print(f"실패 {file_path}: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Embedding 생성 완료: {updated_count}건")
        return updated_count

    # Task 실행 순서: 태그 생성 후 임베딩 생성
    tags_result = generate_search_tags()
    embedding_result = generate_filename_embeddings()
    
    tags_result >> embedding_result