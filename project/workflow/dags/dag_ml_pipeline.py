from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dag_ml_pipeline',
    description='인공지능 모델 학습 파이프라인 - 정기 스케줄링 기반 자동화',
    start_date=pendulum.datetime(2025, 1, 1),
    schedule="@daily",
    tags=['ml_pipeline', 'training', 'automation'],
    default_args=default_args,
    catchup=False  # 과거 실행 누락 방지
) as dag:
    
    @task.virtualenv(
        task_id="train_model_task",
        requirements=[
            'pandas>=2.0.0',
            'scikit-learn>=1.3.0',
            'minio>=7.0.0',
            'polars>=0.19.0',
            'numpy>=1.24.0',
            'pyarrow>=15.0.0'
        ],
        system_site_packages=False  # 격리된 환경 사용
    )
    def train_model_task():
        """
        모델 학습 태스크
        
        Returns:
            dict: 학습 결과 메트릭
        """
        import sys
        import os
        
        # Airflow 컨테이너 내부의 dags 폴더 경로를 sys.path에 추가
        dags_path = '/opt/airflow/dags'
        if dags_path not in sys.path:
            sys.path.insert(0, dags_path)
            
        from scripts.train import train_model
        
        # 설정값 (환경변수나 Airflow Variable로 관리 가능)
        metrics = train_model(
            minio_endpoint='minio:9000',
            minio_access_key='minio',
            minio_secret_key='minio123',
            minio_bucket="raw",
            data_object='csv/Apart_Deal.csv',
            model_bucket="models",
            data_limit=15000,
            test_size=0.2,
            random_state=42,
            n_estimators=100,
            save_model=True
        )
        
        return metrics
    
    # 태스크 실행
    train_result = train_model_task()
