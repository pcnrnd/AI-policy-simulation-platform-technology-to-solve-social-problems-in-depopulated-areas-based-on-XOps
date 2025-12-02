from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_data_integration',
    description='MinIO에서 원본 데이터를 읽어 데이터 통합 후 저장하는 데이터 파이프라인',
    start_date=pendulum.datetime(2025, 11, 27),
    schedule="@daily",
    tags=['data_integration']
) as dag:

    @task.virtualenv(
        task_id="read_minio_data",
        requirements=['minio', 'pandas']
    )
    def read_minio_data():
        from minio import Minio
        from io import BytesIO
        import pandas as pd
        minio_endpoint = 'minio:9000'
        minio_access_key = 'minio'
        minio_secret_key = 'minio123'
        minio_bucket = "raw"
        dfs = []

        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        # 버킷의 모든 객체 나열
        objects = client.list_objects(minio_bucket, recursive=True)
        
        # CSV 파일만 필터링하여 읽기
        for obj in objects:
            object_name = obj.object_name
            # CSV 파일만 처리
            if object_name.endswith('.csv'):
                try:
                    response = client.get_object(minio_bucket, object_name)
                    data_stream = BytesIO(response.read())
                    df_temp = pd.read_csv(data_stream)
                    dfs.append(df_temp)
                    print(f"읽은 파일: {object_name}, 행 수: {len(df_temp)}")
                    response.close()
                    response.release_conn()
                except Exception as e:
                    print(f"파일 읽기 실패 {object_name}: {str(e)}")
                    continue

        if not dfs:
            raise ValueError("읽을 수 있는 CSV 파일이 없습니다.")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"전체 행 수: {len(df)} / 컬럼: {df.columns.tolist()}")
        print(df.head())
        return df


    @task.virtualenv(
        task_id="save_to_minio",
        requirements=['minio', 'pandas']
    )
    def save_to_minio(df):
        from minio import Minio
        from io import BytesIO
        minio_endpoint = 'minio:9000'
        minio_access_key = 'minio'
        minio_secret_key = 'minio123'
        minio_bucket = "prepro"
        minio_object = "prepro_data2.csv"
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        csv_data = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_data)
        client.put_object(
            minio_bucket, 
            minio_object,
            csv_buffer,
            length=len(csv_data),
            content_type='text/csv'
        )



    minio_data = read_minio_data()
    save_to_minio(minio_data)