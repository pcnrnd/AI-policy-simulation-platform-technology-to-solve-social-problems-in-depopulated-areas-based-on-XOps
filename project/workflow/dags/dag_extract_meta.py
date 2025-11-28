from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_extract_meta',
    description='MinIO에서 메타데이터를 추출하여 PostgreSQL에 저장하는 파이프라인',
    start_date=pendulum.datetime(2025, 11, 27),
    schedule="@daily",
    tags=['metadata_extraction']
) as dag:

    @task.virtualenv(
        task_id="extract_minio_metadata",
        requirements=['minio']  # pandas 제거
    )
    def extract_minio_metadata():
        """
        MinIO 버킷의 모든 파일 메타데이터를 추출하는 함수
        파일 내용을 읽지 않고 객체 정보만 사용하여 빠르게 처리
        
        Returns:
            list: 메타데이터 딕셔너리 리스트
        """
        from minio import Minio
        import mimetypes
        import os
        
        minio_endpoint = 'minio:9000'
        minio_access_key = 'minio'
        minio_secret_key = 'minio123'
        minio_bucket = "raw"
        
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        metadata_list = []
        
        # 버킷의 모든 객체 나열 (파일 내용 읽지 않음)
        objects = client.list_objects(minio_bucket, recursive=True)
        
        for obj in objects:
            object_name = obj.object_name
            file_size = obj.size
            
            # 디렉토리는 제외
            if object_name.endswith('/'):
                continue
            
            # 파일 확장자로 file_type 추출
            _, ext = os.path.splitext(object_name)
            file_type = mimetypes.guess_type(object_name)[0] or (ext[1:] if ext else 'unknown')
            
            metadata = {
                'bucket_name': minio_bucket,
                'file_path': object_name,
                'file_type': file_type,
                'file_size': file_size,
                'search_tags': {}  # 초기값은 빈 객체
            }
            
            metadata_list.append(metadata)
            print(f"메타데이터 추출: {object_name} ({file_size} bytes, {file_type})")
        
        if not metadata_list:
            print("경고: 추출된 메타데이터가 없습니다.")
        
        print(f"총 {len(metadata_list)}개 파일의 메타데이터 추출 완료")
        return metadata_list


    @task.virtualenv(
        task_id="save_metadata_to_postgres",
        requirements=['psycopg2-binary']  # pandas 제거
    )
    def save_metadata_to_postgres(metadata_list):
        """
        추출한 메타데이터를 PostgreSQL에 저장하는 함수
        
        Args:
            metadata_list: 메타데이터 딕셔너리 리스트
        
        Returns:
            bool: 저장 성공 여부
        """
        import psycopg2
        import json
        
        conn = psycopg2.connect(
            host='db',
            port='5432',
            database='meta_population',
            user='postgres',
            password='postgres'
        )
        cursor = conn.cursor()
        
        inserted_count = 0
        skipped_count = 0
        error_count = 0
        
        for metadata in metadata_list:
            try:
                insert_sql = """
                    INSERT INTO asset_metadata 
                    (bucket_name, file_path, file_type, file_size, search_tags, processed_status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (bucket_name, file_path) DO NOTHING;
                """
                cursor.execute(
                    insert_sql, 
                    (
                        metadata['bucket_name'], 
                        metadata['file_path'], 
                        metadata['file_type'], 
                        metadata['file_size'],
                        json.dumps(metadata.get('search_tags', {})),  # JSONB로 저장
                        'PENDING'
                    )
                )
                if cursor.rowcount > 0:
                    inserted_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                error_count += 1
                print(f"메타데이터 삽입 실패 {metadata.get('file_path', 'unknown')}: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"삽입 완료: {inserted_count}건, 건너뜀: {skipped_count}건, 오류: {error_count}건")
        return True

    metadata = extract_minio_metadata()
    save_metadata_to_postgres(metadata)