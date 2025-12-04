from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import polars as pl
import numpy as np
import re
import pickle
import json
import os
import sys
import hashlib
from datetime import datetime
from minio import Minio
from io import BytesIO
from typing import Dict, Tuple, Optional


def get_minio_client(
    endpoint: str = 'minio:9000',
    access_key: str = 'minio',
    secret_key: str = 'minio123',
    secure: bool = False
) -> Minio:
    """
    MinIO 클라이언트를 생성하는 함수
    
    Args:
        endpoint: MinIO 엔드포인트
        access_key: 접근 키
        secret_key: 시크릿 키
        secure: SSL 사용 여부
        
    Returns:
        Minio: MinIO 클라이언트 객체
    """
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )


def extract_phase(apartment_name):
    """
    아파트명에서 차수 정보를 추출하는 함수
    예: '남외푸르지오1차' -> '1차', '남운학성타운' -> None
    """
    if pd.isna(apartment_name):
        return None
    match = re.search(r'(\d+차)$', str(apartment_name))
    return match.group(1) if match else None


def remove_phase(apartment_name):
    """
    아파트명에서 차수 정보를 제거하는 함수
    예: '남외푸르지오1차' -> '남외푸르지오', '남운학성타운' -> '남운학성타운'
    """
    if pd.isna(apartment_name):
        return apartment_name
    return re.sub(r'\d+차$', '', str(apartment_name)).strip()


def split_lot_number(lot):
    """
    지번을 본번과 부번으로 분리하는 함수
    '506-1' -> (506, 1)
    '379' -> (379, 0)
    """
    if pd.isna(lot):
        return 0, 0
    
    lot_str = str(lot).strip()
    if '-' in lot_str:
        parts = lot_str.split('-', 1)
        main = int(parts[0]) if parts[0].isdigit() else 0
        sub = int(parts[1]) if parts[1].isdigit() else 0
        return main, sub
    else:
        main = int(lot_str) if lot_str.isdigit() else 0
        return main, 0


def load_data_from_minio(
    client: Minio,
    bucket: str,
    object_name: str,
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    MinIO에서 데이터를 로드하는 함수
    
    Args:
        client: MinIO 클라이언트
        bucket: 버킷 이름
        object_name: 객체 이름
        limit: 읽을 최대 행 수 (None이면 전체)
        
    Returns:
        pl.DataFrame: 로드된 데이터프레임
    """
    try:
        response = client.get_object(bucket, object_name)
        object_data = BytesIO(response.read())
        object_data.seek(0)
        
        df = pl.read_csv(
            object_data,
            schema_overrides={
                '거래금액': pl.Utf8,
                '층': pl.Utf8
            }
        )
        
        if limit:
            df = df[:limit]
            
        response.close()
        response.release_conn()
        
        return df
    except Exception as e:
        raise Exception(f"MinIO에서 데이터 로드 실패: {str(e)}")


def preprocess_data(df: pl.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    데이터 전처리 함수
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: 전처리된 데이터와 LabelEncoder 딕셔너리
    """
    # 차수 추출
    df = df.with_columns(
        pl.col('아파트').map_elements(extract_phase, return_dtype=pl.Utf8).alias('차수')
    )
    
    # 지번 분리
    df = df.with_columns([
        pl.col('지번').map_elements(
            lambda x: split_lot_number(x)[0] if x else 0,
            return_dtype=pl.Int64
        ).alias('지번_본번'),
        pl.col('지번').map_elements(
            lambda x: split_lot_number(x)[1] if x else 0,
            return_dtype=pl.Int64
        ).alias('지번_부번')
    ])
    
    # 아파트명 정제
    df = df.with_columns(
        pl.col('아파트').map_elements(remove_phase, return_dtype=pl.Utf8).alias('아파트_정제')
    )
    
    # 차수 결측치 처리
    df = df.with_columns(
        pl.col('차수').fill_null('없음')
    )
    
    # 필요한 컬럼만 선택
    n_df = df[['지역코드', '법정동', '아파트_정제', '차수', '지번_본번', '지번_부번', 
               '전용면적', '층', '건축년도', '거래금액']]
    
    # Polars를 Pandas로 변환
    n_df_pd = n_df.to_pandas()
    
    # LabelEncoder 생성 및 적용
    encoders = {}
    le_a = LabelEncoder()
    le_b = LabelEncoder()
    le_c = LabelEncoder()
    
    n_df_pd['법정동'] = le_a.fit_transform(n_df_pd['법정동'])
    n_df_pd['차수'] = le_b.fit_transform(n_df_pd['차수'])
    n_df_pd['아파트_정제'] = le_c.fit_transform(n_df_pd['아파트_정제'])
    
    encoders['법정동'] = le_a
    encoders['차수'] = le_b
    encoders['아파트_정제'] = le_c
    
    # 데이터 타입 변환 및 결측치 처리
    n_df_pd['지번_본번'] = n_df_pd['지번_본번'].fillna(0).astype(int)
    n_df_pd['지번_부번'] = n_df_pd['지번_부번'].fillna(0).astype(int)
    n_df_pd['건축년도'] = n_df_pd['건축년도'].fillna(0).astype(int)
    n_df_pd['지역코드'] = n_df_pd['지역코드'].fillna(0).astype(int)
    
    n_df_pd['층'] = pd.to_numeric(
        n_df_pd['층'].replace(' ', np.nan).replace('', np.nan),
        errors='coerce'
    ).fillna(0).astype(int)
    
    n_df_pd['거래금액'] = pd.to_numeric(
        n_df_pd['거래금액'].astype(str).str.replace(',', ''),
        errors='coerce'
    ).fillna(0).astype(int)
    
    return n_df_pd, encoders


def save_artifacts_to_minio(
    client: Minio,
    bucket: str,
    model: RandomForestRegressor,
    encoders: Dict[str, LabelEncoder],
    metrics: Dict[str, float],
    model_name: str = "apartment-price-prediction",
    version: Optional[str] = None,
    hyperparameters: Optional[Dict] = None,
    data_info: Optional[Dict] = None
) -> str:
    """
    모델과 전처리 객체를 MinIO에 저장하는 함수 (Phase 1: 필수 항목 적용)
    
    Args:
        client: MinIO 클라이언트
        bucket: 버킷 이름
        model: 학습된 모델
        encoders: LabelEncoder 딕셔너리
        metrics: 평가 메트릭
        model_name: 모델 이름 (기본값: "apartment-price-prediction")
        version: 모델 버전 (None이면 자동 생성)
        hyperparameters: 하이퍼파라미터 딕셔너리
        data_info: 데이터 정보 딕셔너리
        
    Returns:
        str: 저장된 모델 경로
    """
    try:
        # 버킷이 존재하는지 확인하고 없으면 생성
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
            print(f"✅ 버킷 생성 완료: {bucket}")
        else:
            print(f"✅ 버킷 이미 존재: {bucket}")
        
        # 1. 버전 관리 (필수)
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v1.0.0_{timestamp}"
        
        # 4. 구조화된 경로 (필수): {model_name}/{version}/
        base_path = f"{model_name}/{version}"
        
        # 2. 메타데이터 생성 (필수)
        metadata = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "model_type": type(model).__name__,
            "hyperparameters": hyperparameters or {},
            "data_info": data_info or {},
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "git_commit": os.getenv("GIT_COMMIT", "unknown")
        }
        
        # 파일 저장 헬퍼 함수
        def save_file(file_name: str, data: bytes, content_type: str = 'application/octet-stream'):
            """MinIO에 파일을 저장하는 헬퍼 함수"""
            client.put_object(
                bucket,
                f"{base_path}/{file_name}",
                BytesIO(data),
                length=len(data),
                content_type=content_type
            )
        
        # 모델 저장
        model_buffer = BytesIO()
        pickle.dump(model, model_buffer)
        model_data = model_buffer.getvalue()
        save_file("model.pkl", model_data)
        
        # 인코더 저장
        encoders_buffer = BytesIO()
        pickle.dump(encoders, encoders_buffer)
        save_file("encoders.pkl", encoders_buffer.getvalue())
        
        # 메트릭 저장 (JSON 형태로)
        metrics_json = json.dumps(metrics, indent=2, ensure_ascii=False).encode('utf-8')
        save_file("metrics.json", metrics_json, 'application/json')
        
        # 메타데이터 저장
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False).encode('utf-8')
        save_file("metadata.json", metadata_json, 'application/json')
        
        # 3. 체크섬 저장 (필수)
        model_hash = hashlib.sha256(model_data).hexdigest()
        checksum = {"model_sha256": model_hash}
        checksum_json = json.dumps(checksum).encode('utf-8')
        save_file("checksum.json", checksum_json, 'application/json')
        
        print(f"✅ 모델 저장 완료: {bucket}/{base_path}/")
        print(f"   버전: {version}")
        print(f"   메트릭: R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):,.0f}")
        
        return f"{bucket}/{base_path}"
        
    except Exception as e:
        raise Exception(f"MinIO에 아티팩트 저장 실패: {str(e)}")


def save_artifacts_with_mlflow(
    mlflow_tracking_uri: str,
    model: RandomForestRegressor,
    encoders: Dict[str, LabelEncoder],
    metrics: Dict[str, float],
    hyperparameters: Dict,
    data_info: Dict,
    model_name: str = "apartment-price-prediction",
    experiment_name: str = "apartment-price-prediction",
    use_mlflow: bool = True,
    fallback_to_minio: bool = True,
    minio_client: Optional[Minio] = None,
    minio_bucket: Optional[str] = None
) -> Dict[str, str]:
    """
    MLflow를 사용하여 모델 저장 (하이브리드: MLflow 실패 시 MinIO로 폴백)
    
    Args:
        mlflow_tracking_uri: MLflow 서버 URI
        model: 학습된 모델
        encoders: LabelEncoder 딕셔너리
        metrics: 평가 메트릭
        hyperparameters: 하이퍼파라미터 딕셔너리
        data_info: 데이터 정보 딕셔너리
        model_name: 모델 이름
        experiment_name: 실험 이름
        use_mlflow: MLflow 사용 여부
        fallback_to_minio: MLflow 실패 시 MinIO 폴백 여부
        minio_client: MinIO 클라이언트 (폴백용)
        minio_bucket: MinIO 버킷 이름 (폴백용)
        
    Returns:
        Dict: 저장 정보 (run_id, model_uri, version 등)
    """
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
        import tempfile
        
        if use_mlflow:
            # MLflow 서버 연결 대기 (최대 30초)
            import requests
            import time
            max_retries = 30
            retry_interval = 2
            
            for i in range(max_retries):
                try:
                    response = requests.get(f"{mlflow_tracking_uri}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ MLflow 서버 연결 성공")
                        break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"⏳ MLflow 서버 연결 대기 중... ({i+1}/{max_retries})")
                        time.sleep(retry_interval)
                    else:
                        raise ConnectionError(f"MLflow 서버에 연결할 수 없습니다: {mlflow_tracking_uri}")
            
            # MLflow 설정
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # 실험 생성 또는 가져오기
            # client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            # try:
            #     # 실험 존재 여부 확인
            #     experiment = client.get_experiment_by_name(experiment_name)
            #     if experiment is None:
            #         # 실험이 없으면 생성
            #         print(f"📝 실험 '{experiment_name}' 생성 중...")
            #         experiment_id = client.create_experiment(experiment_name)
            #         print(f"✅ 실험 생성 완료 (ID: {experiment_id})")
            #     else:
            #         print(f"✅ 실험 '{experiment_name}' 존재 확인 (ID: {experiment.experiment_id})")
            # except Exception as e:
            #     # 실험 조회 실패 시 set_experiment로 시도 (자동 생성)
            #     print(f"⚠️ 실험 조회 실패, 자동 생성 시도: {str(e)}")
            #     mlflow.set_experiment(experiment_name)
            # else:
            #     # 실험 설정
            #     mlflow.set_experiment(experiment_name)
          
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run() as run:
                # 하이퍼파라미터 로깅
                mlflow.log_params(hyperparameters)
                
                # 메트릭 로깅
                mlflow.log_metrics(metrics)
                
                # 데이터 정보 로깅
                mlflow.log_params({
                    f"data_{k}": str(v) for k, v in data_info.items()
                })
                
                # 인코더 저장 (artifacts)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    pickle.dump(encoders, tmp_file)
                    tmp_file.flush()
                    mlflow.log_artifact(tmp_file.name, "encoders")
                    os.unlink(tmp_file.name)
                
                # 모델 저장 및 등록
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_model_name = f"{model_name}_{timestamp}"
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=unique_model_name
                )
                
                # 메타데이터 저장
                metadata = {
                    "model_name": unique_model_name,
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "created_at": datetime.now().isoformat(),
                    "data_info": data_info
                }
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_file:
                    json.dump(metadata, tmp_file, indent=2, ensure_ascii=False)
                    tmp_file.flush()
                    mlflow.log_artifact(tmp_file.name, "metadata")
                    os.unlink(tmp_file.name)
                
                # 모델 등록 정보 가져오기
                client = MlflowClient(tracking_uri=mlflow_tracking_uri)
                model_versions = client.get_latest_versions(model_name, stages=["None"])
                if model_versions:
                    model_version = model_versions[0]
                else:
                    # 모델이 등록되지 않은 경우 (이론적으로는 발생하지 않아야 함)
                    model_version = None
                
                result = {
                    "run_id": run.info.run_id,
                    "model_uri": f"runs:/{run.info.run_id}/model",
                    "model_version": model_version.version if model_version else "unknown",
                    "experiment_id": run.info.experiment_id,
                    "storage_type": "mlflow"
                }
                
                print(f"✅ MLflow 저장 완료")
                print(f"   Run ID: {run.info.run_id}")
                if model_version:
                    print(f"   Model Version: {model_version.version}")
                print(f"   Model URI: {result['model_uri']}")
                
                return result
                
    except Exception as e:
        print(f"⚠️ MLflow 저장 실패: {str(e)}")
        if fallback_to_minio and minio_client and minio_bucket:
            print("📦 MinIO로 폴백 저장 중...")
            model_path = save_artifacts_to_minio(
                client=minio_client,
                bucket=minio_bucket,
                model=model,
                encoders=encoders,
                metrics=metrics,
                model_name=model_name,
                hyperparameters=hyperparameters,
                data_info=data_info
            )
            result = {
                "model_path": model_path,
                "storage_type": "minio_fallback",
                "error": str(e)
            }
            return result
        else:
            raise


def train_model(
    minio_endpoint: str = 'minio:9000',
    minio_access_key: str = 'minio',
    minio_secret_key: str = 'minio123',
    minio_bucket: str = "raw",
    data_object: str = 'Apart_Deal.csv',
    model_bucket: str = "models",
    mlflow_tracking_uri: Optional[str] = None,
    use_mlflow: bool = False,
    data_limit: Optional[int] = 15000,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    save_model: bool = True
) -> Dict[str, float]:
    """
    모델 학습 메인 함수
    
    Args:
        minio_endpoint: MinIO 엔드포인트
        minio_access_key: MinIO 접근 키
        minio_secret_key: MinIO 시크릿 키
        minio_bucket: 데이터 버킷 이름
        data_object: 데이터 객체 이름
        model_bucket: 모델 저장 버킷 이름
        mlflow_tracking_uri: MLflow 서버 URI (예: "http://mlflow:5000")
        use_mlflow: MLflow 사용 여부
        data_limit: 데이터 제한 행 수
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드
        n_estimators: 랜덤 포레스트 트리 개수
        save_model: 모델 저장 여부
        
    Returns:
        Dict[str, float]: 평가 메트릭 딕셔너리
    """
    try:
        # MinIO 클라이언트 생성
        client = get_minio_client(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key
        )
        
        # 데이터 로드
        print(f"데이터 로드 중: {minio_bucket}/{data_object}")
        df = load_data_from_minio(client, minio_bucket, data_object, limit=data_limit)
        print(f"로드된 데이터 행 수: {len(df)}")
        
        # 데이터 전처리
        print("데이터 전처리 중...")
        n_df_pd, encoders = preprocess_data(df)
        print(f"전처리 완료. 컬럼 수: {len(n_df_pd.columns)}")
        
        # 타겟과 피처 분리
        y = n_df_pd['거래금액']
        X = n_df_pd.drop(columns=['거래금액'])
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"학습 데이터: {len(X_train)}행, 테스트 데이터: {len(X_test)}행")
        
        # 모델 학습
        print("모델 학습 중...")
        rfc = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rfc.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = rfc.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 거래금액 통계
        y_mean = y_test.mean()
        y_std = y_test.std()
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'y_mean': float(y_mean),
            'y_std': float(y_std),
            'y_min': float(y_test.min()),
            'y_max': float(y_test.max()),
            'rmse_percentage': float((rmse/y_mean)*100),
            'mae_percentage': float((mae/y_mean)*100)
        }
        
        # 결과 출력
        print("=" * 50)
        print("모델 성능 평가 지표")
        print("=" * 50)
        print(f"MSE (Mean Squared Error): {mse:,.2f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:,.2f} 만원")
        print(f"MAE (Mean Absolute Error): {mae:,.2f} 만원")
        print(f"R² Score: {r2:.4f}")
        print()
        print("=" * 50)
        print("실제 거래금액 통계")
        print("=" * 50)
        print(f"평균: {y_mean:,.2f} 만원")
        print(f"표준편차: {y_std:,.2f} 만원")
        print(f"최소값: {y_test.min():,} 만원")
        print(f"최대값: {y_test.max():,} 만원")
        print()
        print("=" * 50)
        print("상대적 성능")
        print("=" * 50)
        print(f"RMSE / 평균: {(rmse/y_mean)*100:.2f}%")
        print(f"MAE / 평균: {(mae/y_mean)*100:.2f}%")
        print(f"R² Score: {r2:.4f} ({r2*100:.2f}% 설명력)")
        
        # 모델 저장
        if save_model:
            print("\n모델 저장 중...")
            
            hyperparameters = {
                "n_estimators": n_estimators,
                "random_state": random_state,
                "test_size": test_size
            }
            
            data_info = {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X.columns),
                "data_limit": data_limit
            }
            
            if use_mlflow and mlflow_tracking_uri:
                # MLflow 사용 (하이브리드: 실패 시 MinIO 폴백)
                storage_info = save_artifacts_with_mlflow(
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    model=rfc,
                    encoders=encoders,
                    metrics=metrics,
                    hyperparameters=hyperparameters,
                    data_info=data_info,
                    model_name="apartment-price-prediction",
                    experiment_name="apartment-price-prediction",
                    use_mlflow=True,
                    fallback_to_minio=True,
                    minio_client=client,
                    minio_bucket=model_bucket
                )
                print(f"📦 저장 정보: {storage_info}")
            else:
                # MinIO 사용 (기존 방식)
                model_path = save_artifacts_to_minio(
                    client=client,
                    bucket=model_bucket,
                    model=rfc,
                    encoders=encoders,
                    metrics=metrics,
                    model_name="apartment-price-prediction",
                    version=None,  # 자동 생성
                    hyperparameters=hyperparameters,
                    data_info=data_info
                )
                print(f"📦 모델 저장 경로: {model_path}")
        
        return metrics
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        raise


# 스크립트 직접 실행 시
if __name__ == "__main__":
    train_model()
