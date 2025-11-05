from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dags_data_integration',
    description='MinIO에서 원본 데이터를 읽어 전처리 후 저장하는 데이터 파이프라인',
    start_date=pendulum.datetime(2025, 11, 4),
    schedule="@daily",
    tags=['dataops', 'minio', 'integration']
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
        # minio_bucket = 'raw'
        # minio_object = 'restaurant_2021.csv'

        minio_bucket = "raw"
        # minio_object = "restaurant"
        years = ['2020', '2021', '2022', '2023', '2024', '2025']
        dfs = []

        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        for year in years:
            object_name = f"restaurant_{year}.csv"
            response = client.get_object(minio_bucket, object_name)
            data_stream = BytesIO(response.read())
            df_temp = pd.read_csv(data_stream)
            dfs.append(df_temp)
            response.close()
            response.release_conn()

        df = pd.concat(dfs, ignore_index=True)
        print(f"row: {len(df)} / columns{df.columns}")
        print(df.head())
        # response = client.get_object(minio_bucket, minio_object)
        # df = pd.read_csv(BytesIO(response.read()))
        # print(len(df))
        # print(df.head())
        # response.close()
        # response.release_conn()
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
        minio_object = "prepro_data.csv"
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