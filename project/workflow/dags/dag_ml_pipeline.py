from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta
from scripts.train import train_model

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
    default_args=default_args
) as dag:
    @task.virtualenv(
        task_id="load_and_validate_data",
        requirements=['pandas', 'scikit-learn']
    )
    def load_and_validate_data():
        from sklearn.datasets import load_iris
        import pandas as pd

        iris = load_iris(as_frame=True)
        df = iris['frame']  # feature와 target이 모두 포함된 DataFrame
        print(df.head())
        return df

    @task.virtualenv(
        task_id="preprocess_data",
        requirements=['pandas', 'scikit-learn']
    )
    def preprocess_data(df):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
        print(df.head())
        return df

    @task.virtualenv(
        task_id="train_model",
        requirements=['pandas', 'scikit-learn']
    )
    def train_model(df):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']], df['target']) 
        return model

    load_and_validate_data() >> preprocess_data() >> train_model()