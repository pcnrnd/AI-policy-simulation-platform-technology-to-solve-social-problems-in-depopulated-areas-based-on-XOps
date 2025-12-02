from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_data_version_management',
    description='데이터 버전 관리 데이터 파이프라인',
    start_date=pendulum.datetime(2025, 11, 4),
    schedule="@daily",
    tags=['version_management'],
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
        task_id="save_to_dvc",
        requirements=['dvc', 'pandas', 'pyarrow', 'dvc-s3']
    )
    def save_to_dvc(df):
        import subprocess
        import pandas as pd
        from pathlib import Path
        import os
        import sys
        
        def run_dvc_command(cmd, check=True, capture_output=True):
            """DVC 명령어 실행 헬퍼 함수 (virtualenv 환경에서 실행 가능하도록 수정)"""
            # virtualenv 환경에서 dvc 명령어를 찾기 위해 python -m dvc 사용
            dvc_cmd = [sys.executable, '-m', 'dvc'] + cmd
            result = subprocess.run(
                dvc_cmd,
                check=check,
                capture_output=capture_output,
                text=True
            )
            if result.returncode != 0 and check:
                raise RuntimeError(
                    f"DVC command failed: {' '.join(dvc_cmd)}\n"
                    f"STDERR: {result.stderr}\n"
                    f"STDOUT: {result.stdout}"
                )
            return result
        
        # DVC 작업 디렉토리 설정
        dvc_dir = Path('/opt/airflow/data')
        dvc_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(dvc_dir)
        
        # Git 저장소 초기화 (DVC는 Git이 필요함)
        if not (dvc_dir / '.git').exists():
            print("Initializing Git repository...")
            subprocess.run(['git', 'init'], check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 'jwlee2301@pcninc.co.kr'], check=False)
            subprocess.run(['git', 'config', 'user.name', 'jwlee'], check=False)
        
        # DVC 저장소 초기화 (없을 때만)
        if not (dvc_dir / '.dvc').exists():
            print("Initializing DVC repository...")
            run_dvc_command(['init'], check=False)
        
        # 원격 저장소 설정 (없을 때만 추가)
        remote_name = 'minio_dvc'
        # virtualenv 환경에서 dvc 실행을 위해 python -m dvc 사용
        remote_list_result = subprocess.run(
            [sys.executable, '-m', 'dvc', 'remote', 'list'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if remote_name not in remote_list_result.stdout:
            print(f"Setting up DVC remote: {remote_name}")
            # 원격 저장소 추가 (이미 존재하면 에러 무시)
            add_result = run_dvc_command(['remote', 'add', remote_name, 's3://dvc'], check=False)
            if add_result.returncode != 0 and 'already exists' not in add_result.stderr.lower():
                print(f"Warning: Remote add returned: {add_result.stderr}")
            
            # 원격 저장소 설정
            run_dvc_command(['remote', 'modify', remote_name, 'endpointurl', 'http://minio:9000'], check=False)
            run_dvc_command(['remote', 'modify', remote_name, 'access_key_id', 'minio'], check=False)
            run_dvc_command(['remote', 'modify', remote_name, 'secret_access_key', 'minio123'], check=False)
            # 기본 원격으로 설정
            run_dvc_command(['remote', 'default', remote_name], check=False)
            print(f"Remote '{remote_name}' configured and set as default")
        else:
            print(f"Remote '{remote_name}' already exists")
        
        # DataFrame을 파일로 저장 (Parquet - 압축 효율 좋음)
        output_file = dvc_dir / 'data' / 'raw_data.parquet'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"Data saved to file: {output_file}")
        
        # DVC에 파일 추가 (이미 추적 중이면 강제 업데이트)
        dvc_file = Path(str(output_file) + '.dvc')
        force_flag = ['-f'] if dvc_file.exists() else []
        
        result = run_dvc_command(['add'] + force_flag + [str(output_file)])
        print(f"DVC add completed: {output_file}")
        
        # 원격 저장소 확인
        remote_check = subprocess.run(
            [sys.executable, '-m', 'dvc', 'remote', 'default'],
            capture_output=True,
            text=True,
            check=False
        )
        default_remote = remote_check.stdout.strip() if remote_check.returncode == 0 else remote_name
        print(f"Using remote: {default_remote or remote_name}")
        
        # DVC push (명시적으로 원격 저장소 지정)
        print(f"Pushing to remote: {remote_name}")
        push_result = run_dvc_command(['push', '-r', remote_name], check=False)
        
        # 에러 상세 출력
        if push_result.returncode != 0:
            print(f"Push failed with return code: {push_result.returncode}")
            print(f"STDOUT: {push_result.stdout}")
            print(f"STDERR: {push_result.stderr}")
            # 에러가 발생했지만 데이터는 저장되었으므로 경고만 출력
            print(f"WARNING: Push failed, but data is saved locally: {output_file}")
        else:
            if push_result.stdout:
                print(f"Push output: {push_result.stdout}")
            if push_result.stderr:
                print(f"Push warnings: {push_result.stderr}")
            print(f"Successfully pushed to remote: {remote_name}")
        
        print(f"Data successfully saved to DVC: {output_file}")
        return df
    

    raw_data = read_minio_data()
    save_to_dvc(raw_data)
