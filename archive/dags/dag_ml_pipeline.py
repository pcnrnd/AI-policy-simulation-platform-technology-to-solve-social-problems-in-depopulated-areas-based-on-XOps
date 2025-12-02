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
        requirements=['minio', 'pandas', 'numpy']
    )
    def load_and_validate_data():
        """
        MinIO에서 전처리된 데이터를 로드하고 데이터 품질을 검증하는 함수
        
        Returns:
            dict: 검증된 데이터와 메타데이터를 포함한 딕셔너리
        """
        from minio import Minio
        from io import BytesIO
        import pandas as pd
        import numpy as np
        
        minio_endpoint = 'minio:9000'
        minio_access_key = 'minio'
        minio_secret_key = 'minio123'
        minio_bucket = "prepro"
        minio_object = "prepro_data.csv"
        
        # MinIO 클라이언트 생성
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        # 데이터 로드
        try:
            response = client.get_object(minio_bucket, minio_object)
            data_stream = BytesIO(response.read())
            df = pd.read_csv(data_stream)
            response.close()
            response.release_conn()
            print(f"데이터 로드 완료: {len(df)}행, {len(df.columns)}열")
        except Exception as e:
            raise ValueError(f"데이터 로드 실패: {str(e)}")
        
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
        required_columns = []  # 필요시 필수 컬럼 리스트 추가
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        print(f"데이터 검증 완료:")
        print(f"  - 총 행 수: {validation_results['total_rows']}")
        print(f"  - 총 열 수: {validation_results['total_columns']}")
        print(f"  - 중복 행: {validation_results['duplicate_rows']}")
        print(f"  - 결측치 비율: {missing_ratio:.2%}")
        
        # 데이터 분할 정보 저장 (나중에 사용)
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        return {
            'data': df.to_dict('records'),  # JSON 직렬화를 위해 dict로 변환
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
        requirements=['mlflow', 'pandas', 'numpy', 'scikit-learn', 'boto3']
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
        
        # MLflow 설정
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        mlflow.set_experiment("population_decline_prediction")
        
        # 데이터 복원
        df = pd.DataFrame(data_info['data'])
        columns = data_info['columns']
        df = df[columns]  # 컬럼 순서 보장
        
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
                'test_data': {
                    'X_test': X_test.to_dict('records'),
                    'y_test': y_test.tolist(),
                    'columns': X_test.columns.tolist()
                }
            }

    @task.virtualenv(
        task_id="evaluate_model",
        requirements=['mlflow', 'pandas', 'numpy', 'scikit-learn']
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
        
        # MLflow 설정
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        
        # 테스트 데이터 복원
        test_data = training_result['test_data']
        X_test = pd.DataFrame(test_data['X_test'])
        X_test = X_test[test_data['columns']]  # 컬럼 순서 보장
        y_test = np.array(test_data['y_test'])
        
        # 모델 로드
        run_id = training_result['run_id']
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
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
        requirements=['mlflow', 'boto3']
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
        
        # MLflow 설정
        mlflow.set_tracking_uri("http://mlflow-server:5000")
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
