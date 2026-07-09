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
    schedule="@daily",  # 매일 새벽 실행
    tags=['ml_pipeline', 'training', 'automation'],
    catchup=False
) as dag:

    @task.virtualenv(
        task_id="load_and_validate_data",
        requirements=['pandas', 'numpy']  # minio 제거
    )
    def load_and_validate_data():
        """
        더미 데이터를 생성하고 데이터 품질을 검증하는 함수
        
        Returns:
            dict: 검증된 데이터와 메타데이터를 포함한 딕셔너리
        """
        import pandas as pd
        import numpy as np
        
        # 더미 데이터 생성 설정
        n_samples = 1000
        np.random.seed(42)
        
        # 인구 감소 예측을 위한 더미 데이터 생성
        # 피처: 지역별 인구 통계, 경제 지표, 사회 지표 등
        data = {
            # 지역 정보
            'region_id': np.random.randint(1, 21, n_samples),  # 20개 지역
            'urban_ratio': np.random.uniform(0.3, 0.9, n_samples),  # 도시화 비율
            
            # 인구 통계
            'total_population': np.random.randint(50000, 500000, n_samples),  # 총 인구
            'population_density': np.random.uniform(100, 5000, n_samples),  # 인구 밀도
            'age_median': np.random.uniform(30, 50, n_samples),  # 중위 연령
            'elderly_ratio': np.random.uniform(0.1, 0.3, n_samples),  # 고령자 비율
            'youth_ratio': np.random.uniform(0.15, 0.35, n_samples),  # 청년 비율
            
            # 경제 지표
            'gdp_per_capita': np.random.uniform(20000, 60000, n_samples),  # 1인당 GDP
            'unemployment_rate': np.random.uniform(0.02, 0.12, n_samples),  # 실업률
            'income_median': np.random.uniform(30000, 70000, n_samples),  # 중위 소득
            
            # 사회 지표
            'education_level': np.random.uniform(0.5, 0.95, n_samples),  # 고등교육 이수율
            'marriage_rate': np.random.uniform(0.4, 0.8, n_samples),  # 결혼율
            'birth_rate': np.random.uniform(0.5, 1.5, n_samples),  # 출생률
            
            # 인프라 지표
            'medical_facilities': np.random.uniform(0.5, 3.0, n_samples),  # 의료시설 밀도
            'school_density': np.random.uniform(0.3, 2.0, n_samples),  # 학교 밀도
            'transport_accessibility': np.random.uniform(0.4, 0.95, n_samples),  # 교통 접근성
        }
        
        # 타겟 변수 생성 (인구 감소율)
        # 실제 관계를 모방하기 위해 여러 피처의 조합으로 생성
        population_decline_rate = (
            -0.001 * data['elderly_ratio'] +
            -0.0005 * data['unemployment_rate'] +
            -0.0003 * (1 - data['birth_rate']) +
            -0.0002 * (1 - data['education_level']) +
            -0.0001 * (1 - data['marriage_rate']) +
            np.random.normal(0, 0.002, n_samples)  # 노이즈 추가
        )
        
        # 음수 값 제거 (감소율이므로 음수)
        population_decline_rate = np.clip(population_decline_rate, -0.05, 0.01)
        data['population_decline_rate'] = population_decline_rate
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 소량의 결측치 추가 (실제 데이터를 모방)
        missing_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
        for idx in missing_indices:
            col = np.random.choice(df.columns[:-1])  # 타겟 변수 제외
            df.loc[idx, col] = np.nan
        
        print(f"더미 데이터 생성 완료: {len(df)}행, {len(df.columns)}열")
        
        # 데이터 품질 검증
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # 결측치 비율 확인
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.5:
            raise ValueError(f"결측치 비율이 너무 높습니다: {missing_ratio:.2%}")
        
        # 데이터가 비어있는지 확인
        if len(df) == 0:
            raise ValueError("데이터가 비어있습니다.")
        
        # 스키마 검증 (기본 컬럼 존재 여부 확인)
        required_columns = ['population_decline_rate']  # 타겟 변수는 필수
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        print(f"데이터 검증 완료:")
        print(f"  - 총 행 수: {validation_results['total_rows']}")
        print(f"  - 총 열 수: {validation_results['total_columns']}")
        print(f"  - 중복 행: {validation_results['duplicate_rows']}")
        print(f"  - 결측치 비율: {missing_ratio:.2%}")
        print(f"  - 타겟 변수 범위: {df['population_decline_rate'].min():.4f} ~ {df['population_decline_rate'].max():.4f}")
        
        # 데이터 분할 정보 저장 (나중에 사용)
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        # XCom 크기 제한을 피하기 위해 메타데이터만 반환
        # 다음 태스크에서 동일한 시드로 데이터 재생성
        return {
            'n_samples': n_samples,
            'random_seed': 42,
            'columns': df.columns.tolist(),
            'validation_results': validation_results,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }

    @task.virtualenv(
        task_id="train_model",
        requirements=['mlflow', 'pandas', 'numpy', 'scikit-learn', 'boto3', 'requests']
    )
    def train_model(data_info):
        """
        MLflow를 사용하여 모델을 학습하는 함수
        
        Args:
            data_info: load_and_validate_data에서 반환된 데이터 정보
            
        Returns:
            dict: 학습된 모델 정보 및 메트릭
        """
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import os
        import time
        import requests
        
        # MLflow 서버 연결 대기 함수
        def wait_for_mlflow_server(mlflow_uri="http://mlflow-server:5000", max_retries=30, retry_interval=2):
            """MLflow 서버가 준비될 때까지 대기"""
            for i in range(max_retries):
                try:
                    response = requests.get(f"{mlflow_uri}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"MLflow 서버 연결 성공")
                        return True
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"MLflow 서버 연결 대기 중... ({i+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_interval)
                    else:
                        raise ConnectionError(f"MLflow 서버에 연결할 수 없습니다: {mlflow_uri}")
            return False
        
        # MLflow 서버 연결 대기
        mlflow_uri = "http://mlflow-server:5001"
        wait_for_mlflow_server(mlflow_uri)
        
        # MLflow 설정
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("population_decline_prediction")
        
        # 데이터 재생성 (XCom 크기 제한을 피하기 위해)
        n_samples = data_info['n_samples']
        random_seed = data_info['random_seed']
        np.random.seed(random_seed)
        
        # 인구 감소 예측을 위한 더미 데이터 생성
        data = {
            # 지역 정보
            'region_id': np.random.randint(1, 21, n_samples),  # 20개 지역
            'urban_ratio': np.random.uniform(0.3, 0.9, n_samples),  # 도시화 비율
            
            # 인구 통계
            'total_population': np.random.randint(50000, 500000, n_samples),  # 총 인구
            'population_density': np.random.uniform(100, 5000, n_samples),  # 인구 밀도
            'age_median': np.random.uniform(30, 50, n_samples),  # 중위 연령
            'elderly_ratio': np.random.uniform(0.1, 0.3, n_samples),  # 고령자 비율
            'youth_ratio': np.random.uniform(0.15, 0.35, n_samples),  # 청년 비율
            
            # 경제 지표
            'gdp_per_capita': np.random.uniform(20000, 60000, n_samples),  # 1인당 GDP
            'unemployment_rate': np.random.uniform(0.02, 0.12, n_samples),  # 실업률
            'income_median': np.random.uniform(30000, 70000, n_samples),  # 중위 소득
            
            # 사회 지표
            'education_level': np.random.uniform(0.5, 0.95, n_samples),  # 고등교육 이수율
            'marriage_rate': np.random.uniform(0.4, 0.8, n_samples),  # 결혼율
            'birth_rate': np.random.uniform(0.5, 1.5, n_samples),  # 출생률
            
            # 인프라 지표
            'medical_facilities': np.random.uniform(0.5, 3.0, n_samples),  # 의료시설 밀도
            'school_density': np.random.uniform(0.3, 2.0, n_samples),  # 학교 밀도
            'transport_accessibility': np.random.uniform(0.4, 0.95, n_samples),  # 교통 접근성
        }
        
        # 타겟 변수 생성 (인구 감소율)
        population_decline_rate = (
            -0.001 * data['elderly_ratio'] +
            -0.0005 * data['unemployment_rate'] +
            -0.0003 * (1 - data['birth_rate']) +
            -0.0002 * (1 - data['education_level']) +
            -0.0001 * (1 - data['marriage_rate']) +
            np.random.normal(0, 0.002, n_samples)  # 노이즈 추가
        )
        
        # 음수 값 제거 (감소율이므로 음수)
        population_decline_rate = np.clip(population_decline_rate, -0.05, 0.01)
        data['population_decline_rate'] = population_decline_rate
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 소량의 결측치 추가 (실제 데이터를 모방)
        missing_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
        for idx in missing_indices:
            col = np.random.choice(df.columns[:-1])  # 타겟 변수 제외
            df.loc[idx, col] = np.nan
        
        # 컬럼 순서 보장
        columns = data_info['columns']
        df = df[columns]
        
        # 데이터 분할
        split_ratios = data_info['split_ratios']
        train_ratio = split_ratios['train']
        val_ratio = split_ratios['val']
        
        # 타겟 변수 설정 (예시: 마지막 컬럼을 타겟으로 가정)
        # 실제 프로젝트에 맞게 수정 필요
        if len(df.columns) < 2:
            raise ValueError("타겟 변수를 설정하기에 충분한 컬럼이 없습니다.")
        
        # 마지막 컬럼을 타겟으로, 나머지를 피처로 설정
        feature_columns = df.columns[:-1].tolist()
        target_column = df.columns[-1]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # 학습/검증/테스트 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=split_ratios['test'], random_state=42
        )
        
        val_size = split_ratios['val'] / (split_ratios['train'] + split_ratios['val'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )
        
        print(f"데이터 분할 완료:")
        print(f"  - 학습: {len(X_train)}행")
        print(f"  - 검증: {len(X_val)}행")
        print(f"  - 테스트: {len(X_test)}행")
        
        # 하이퍼파라미터 설정
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # MLflow 실험 시작
        with mlflow.start_run(run_name=f"training_{pendulum.now().format('YYYY-MM-DD_HH-mm-ss')}"):
            # 하이퍼파라미터 로깅
            mlflow.log_params(params)
            
            # 모델 학습
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # 메트릭 로깅
            metrics = {
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2
            }
            
            mlflow.log_metrics(metrics)
            
            # 모델 저장
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="PopulationDeclineModel"
            )
            
            # 데이터셋 정보 로깅
            mlflow.log_params({
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'n_features': len(feature_columns),
                'target_column': target_column
            })
            
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow Run ID: {run_id}")
            print(f"학습 메트릭:")
            print(f"  - 학습 R²: {train_r2:.4f}")
            print(f"  - 검증 R²: {val_r2:.4f}")
            print(f"  - 검증 MSE: {val_mse:.4f}")
            print(f"  - 검증 MAE: {val_mae:.4f}")
            
            return {
                'run_id': run_id,
                'metrics': metrics,
                'model_name': 'PopulationDeclineModel',
                'feature_columns': feature_columns,
                'target_column': target_column,
                'n_samples': n_samples,
                'random_seed': random_seed,
                'split_ratios': data_info['split_ratios']
            }

    @task.virtualenv(
        task_id="evaluate_model",
        requirements=['mlflow', 'pandas', 'numpy', 'scikit-learn', 'requests']
    )
    def evaluate_model(training_result):
        """
        테스트 세트를 사용하여 모델을 평가하는 함수
        
        Args:
            training_result: train_model에서 반환된 학습 결과
            
        Returns:
            dict: 평가 메트릭 및 모델 성능 정보
        """
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import time
        import requests
        
        # MLflow 서버 연결 대기 함수
        def wait_for_mlflow_server(mlflow_uri="http://mlflow-server:5000", max_retries=30, retry_interval=2):
            """MLflow 서버가 준비될 때까지 대기"""
            for i in range(max_retries):
                try:
                    response = requests.get(f"{mlflow_uri}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"MLflow 서버 연결 성공")
                        return True
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"MLflow 서버 연결 대기 중... ({i+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_interval)
                    else:
                        raise ConnectionError(f"MLflow 서버에 연결할 수 없습니다: {mlflow_uri}")
            return False
        
        # MLflow 서버 연결 대기
        mlflow_uri = "http://mlflow-server:5000"
        wait_for_mlflow_server(mlflow_uri)
        
        # MLflow 설정
        mlflow.set_tracking_uri(mlflow_uri)
        
        # 모델 로드
        run_id = training_result['run_id']
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 테스트 데이터 재생성 (동일한 시드와 분할 비율 사용)
        n_samples = training_result['n_samples']
        random_seed = training_result['random_seed']
        split_ratios = training_result['split_ratios']
        feature_columns = training_result['feature_columns']
        target_column = training_result['target_column']
        
        np.random.seed(random_seed)
        
        # 인구 감소 예측을 위한 더미 데이터 생성
        data = {
            'region_id': np.random.randint(1, 21, n_samples),
            'urban_ratio': np.random.uniform(0.3, 0.9, n_samples),
            'total_population': np.random.randint(50000, 500000, n_samples),
            'population_density': np.random.uniform(100, 5000, n_samples),
            'age_median': np.random.uniform(30, 50, n_samples),
            'elderly_ratio': np.random.uniform(0.1, 0.3, n_samples),
            'youth_ratio': np.random.uniform(0.15, 0.35, n_samples),
            'gdp_per_capita': np.random.uniform(20000, 60000, n_samples),
            'unemployment_rate': np.random.uniform(0.02, 0.12, n_samples),
            'income_median': np.random.uniform(30000, 70000, n_samples),
            'education_level': np.random.uniform(0.5, 0.95, n_samples),
            'marriage_rate': np.random.uniform(0.4, 0.8, n_samples),
            'birth_rate': np.random.uniform(0.5, 1.5, n_samples),
            'medical_facilities': np.random.uniform(0.5, 3.0, n_samples),
            'school_density': np.random.uniform(0.3, 2.0, n_samples),
            'transport_accessibility': np.random.uniform(0.4, 0.95, n_samples),
        }
        
        # 타겟 변수 생성
        population_decline_rate = (
            -0.001 * data['elderly_ratio'] +
            -0.0005 * data['unemployment_rate'] +
            -0.0003 * (1 - data['birth_rate']) +
            -0.0002 * (1 - data['education_level']) +
            -0.0001 * (1 - data['marriage_rate']) +
            np.random.normal(0, 0.002, n_samples)
        )
        population_decline_rate = np.clip(population_decline_rate, -0.05, 0.01)
        data['population_decline_rate'] = population_decline_rate
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 소량의 결측치 추가
        missing_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
        for idx in missing_indices:
            col = np.random.choice(df.columns[:-1])
            df.loc[idx, col] = np.nan
        
        # 데이터 분할 (동일한 random_state로 분할 보장)
        from sklearn.model_selection import train_test_split
        X = df[feature_columns]
        y = df[target_column]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=split_ratios['test'], random_state=42
        )
        
        # 테스트 세트 예측
        y_test_pred = model.predict(X_test)
        
        # 평가 메트릭 계산
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 성능 임계값 설정 (예시)
        performance_threshold = {
            'min_r2': 0.5,
            'max_mse': float('inf'),
            'max_mae': float('inf')
        }
        
        # 성능 검증
        performance_acceptable = (
            test_r2 >= performance_threshold['min_r2'] and
            test_mse <= performance_threshold['max_mse'] and
            test_mae <= performance_threshold['max_mae']
        )
        
        evaluation_results = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'performance_acceptable': performance_acceptable,
            'thresholds': performance_threshold
        }
        
        # MLflow에 테스트 메트릭 로깅
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2
            })
        
        print(f"모델 평가 완료:")
        print(f"  - 테스트 R²: {test_r2:.4f}")
        print(f"  - 테스트 MSE: {test_mse:.4f}")
        print(f"  - 테스트 MAE: {test_mae:.4f}")
        print(f"  - 성능 기준 충족: {performance_acceptable}")
        
        return {
            'run_id': run_id,
            'evaluation_results': evaluation_results,
            'model_name': training_result['model_name']
        }

    @task.virtualenv(
        task_id="register_and_deploy_model",
        requirements=['mlflow', 'boto3', 'requests']
    )
    def register_and_deploy_model(evaluation_result):
        """
        MLflow Model Registry에 모델을 등록하고 Production으로 승격하는 함수
        
        Args:
            evaluation_result: evaluate_model에서 반환된 평가 결과
            
        Returns:
            dict: 모델 등록 및 배포 정보
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        import time
        import requests
        
        # MLflow 서버 연결 대기 함수
        def wait_for_mlflow_server(mlflow_uri="http://mlflow-server:5000", max_retries=30, retry_interval=2):
            """MLflow 서버가 준비될 때까지 대기"""
            for i in range(max_retries):
                try:
                    response = requests.get(f"{mlflow_uri}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"MLflow 서버 연결 성공")
                        return True
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"MLflow 서버 연결 대기 중... ({i+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_interval)
                    else:
                        raise ConnectionError(f"MLflow 서버에 연결할 수 없습니다: {mlflow_uri}")
            return False
        
        # MLflow 서버 연결 대기
        mlflow_uri = "http://mlflow-server:5000"
        wait_for_mlflow_server(mlflow_uri)
        
        # MLflow 설정
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        run_id = evaluation_result['run_id']
        model_name = evaluation_result['model_name']
        performance_acceptable = evaluation_result['evaluation_results']['performance_acceptable']
        
        # 모델 등록
        try:
            model_version = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name
            )
            print(f"모델 등록 완료: {model_name} (Version {model_version.version})")
        except Exception as e:
            print(f"모델 등록 중 오류 발생: {str(e)}")
            # 이미 등록된 경우 최신 버전 가져오기
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                model_version = latest_versions[0]
                print(f"기존 모델 버전 사용: Version {model_version.version}")
            else:
                raise
        
        # 성능 기준 충족 시 Production으로 승격
        if performance_acceptable:
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production"
                )
                print(f"모델을 Production으로 승격: {model_name} (Version {model_version.version})")
                deployment_status = "deployed"
            except Exception as e:
                print(f"Production 승격 중 오류 발생: {str(e)}")
                deployment_status = "registration_failed"
        else:
            # 성능 기준 미충족 시 Staging으로 등록
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                print(f"모델을 Staging으로 등록: {model_name} (Version {model_version.version})")
                deployment_status = "staging"
            except Exception as e:
                print(f"Staging 등록 중 오류 발생: {str(e)}")
                deployment_status = "registration_failed"
        
        # 모델 정보 반환
        return {
            'model_name': model_name,
            'model_version': model_version.version,
            'run_id': run_id,
            'deployment_status': deployment_status,
            'performance_acceptable': performance_acceptable
        }

    # 파이프라인 실행 순서 정의
    data_info = load_and_validate_data()
    training_result = train_model(data_info)
    evaluation_result = evaluate_model(training_result)
    register_and_deploy_model(evaluation_result)
