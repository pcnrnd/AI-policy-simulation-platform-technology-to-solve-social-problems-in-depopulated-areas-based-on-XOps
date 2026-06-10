from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_restaurant_data_integration',
    description='MinIO에서 restaurant 데이터를 읽어 통합 후 저장하는 데이터 파이프라인',
    start_date=pendulum.datetime(2025, 11, 27),
    schedule="@daily",
    tags=['restaurant_integration'],
    default_args=default_args
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
        file_info = []

        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        # 버킷의 모든 객체 나열
        objects = client.list_objects(minio_bucket, recursive=True)
        
        # Restaurant CSV 파일만 필터링하여 읽기
        restaurant_files = []
        for obj in objects:
            object_name = obj.object_name
            # Restaurant가 포함된 CSV 파일만 처리
            if object_name.endswith('.csv') and 'restaurant' in object_name.lower():
                restaurant_files.append(object_name)
        
        print(f"📋 발견된 restaurant 파일 수: {len(restaurant_files)}")
        print(f"파일 목록: {restaurant_files}\n")
        
        if not restaurant_files:
            raise ValueError("처리할 restaurant CSV 파일이 없습니다.")
        
        # Restaurant 파일만 읽기
        for object_name in restaurant_files:
            try:
                response = client.get_object(minio_bucket, object_name)
                data_stream = BytesIO(response.read())
                df_temp = pd.read_csv(data_stream)
                
                # 파일 정보 저장
                file_info.append({
                    'file_name': object_name,
                    'rows': len(df_temp),
                    'columns': list(df_temp.columns)
                })
                
                # 소스 파일 정보 추가 (선택사항)
                df_temp['source_file'] = object_name
                
                dfs.append(df_temp)
                print(f"✅ 읽은 파일: {object_name}, 행 수: {len(df_temp)}, 컬럼 수: {len(df_temp.columns)}")
                response.close()
                response.release_conn()
            except Exception as e:
                print(f"❌ 파일 읽기 실패 {object_name}: {str(e)}")
                continue

        if not dfs:
            raise ValueError("읽을 수 있는 restaurant CSV 파일이 없습니다.")
        
        # 파일별 정보 출력
        print("\n" + "="*50)
        print("Restaurant 파일별 정보")
        print("="*50)
        for info in file_info:
            print(f"파일: {info['file_name']}")
            print(f"  행 수: {info['rows']}, 컬럼: {info['columns']}\n")
        
        # 모든 데이터프레임의 컬럼 통합 (다른 스키마 대응)
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = sorted(list(all_columns))
        
        print(f"통합 컬럼 수: {len(all_columns)}")
        print(f"통합 컬럼 목록: {all_columns}\n")
        
        # 각 데이터프레임의 컬럼을 통합 컬럼에 맞춤
        aligned_dfs = []
        for i, df in enumerate(dfs):
            df_aligned = df.copy()
            # 누락된 컬럼 추가 (NaN으로 채움)
            for col in all_columns:
                if col not in df_aligned.columns:
                    df_aligned[col] = None
            # 컬럼 순서 정렬
            df_aligned = df_aligned[all_columns]
            aligned_dfs.append(df_aligned)
            print(f"정렬 완료: {file_info[i]['file_name']} -> 컬럼 수: {len(df_aligned.columns)}")
        
        # 데이터프레임 합치기
        try:
            df = pd.concat(aligned_dfs, ignore_index=True)
            print(f"\n✅ Restaurant 데이터 통합 완료!")
            print(f"전체 행 수: {len(df)}")
            print(f"전체 컬럼 수: {len(df.columns)}")
            print(f"컬럼 목록: {list(df.columns)}")
            print(f"\n결측치 정보:")
            print(df.isnull().sum())
            print(f"\n샘플 데이터:")
            print(df.head())
        except Exception as e:
            print(f"❌ 데이터 통합 실패: {str(e)}")
            raise
        
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
        minio_object = "restaurant_integrated.csv"
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        # 버킷이 없으면 생성
        if not client.bucket_exists(minio_bucket):
            client.make_bucket(minio_bucket)
            print(f"✅ 버킷 생성: {minio_bucket}")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_data)
        client.put_object(
            minio_bucket, 
            minio_object,
            csv_buffer,
            length=len(csv_data),
            content_type='text/csv'
        )
        print(f"✅ 저장 완료: {minio_bucket}/{minio_object}")
        print(f"   행 수: {len(df)}, 컬럼 수: {len(df.columns)}")


    minio_data = read_minio_data()
    save_to_minio(minio_data)

