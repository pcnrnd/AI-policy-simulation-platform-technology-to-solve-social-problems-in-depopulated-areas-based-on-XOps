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
    model_name: str = "model.pkl",
    encoders_name: str = "encoders.pkl"
) -> None:
    """
    모델과 전처리 객체를 MinIO에 저장하는 함수
    
    Args:
        client: MinIO 클라이언트
        bucket: 버킷 이름
        model: 학습된 모델
        encoders: LabelEncoder 딕셔너리
        metrics: 평가 메트릭
        model_name: 모델 파일명
        encoders_name: 인코더 파일명
    """
    try:
        # 모델 저장
        model_buffer = BytesIO()
        pickle.dump(model, model_buffer)
        model_buffer.seek(0)
        client.put_object(
            bucket,
            f"models/{model_name}",
            model_buffer,
            length=len(model_buffer.getvalue()),
            content_type='application/octet-stream'
        )
        
        # 인코더 저장
        encoders_buffer = BytesIO()
        pickle.dump(encoders, encoders_buffer)
        encoders_buffer.seek(0)
        client.put_object(
            bucket,
            f"models/{encoders_name}",
            encoders_buffer,
            length=len(encoders_buffer.getvalue()),
            content_type='application/octet-stream'
        )
        
        # 메트릭 저장 (JSON 형태로)
        metrics_buffer = BytesIO(json.dumps(metrics, indent=2).encode('utf-8'))
        metrics_buffer.seek(0)
        client.put_object(
            bucket,
            f"models/metrics.json",
            metrics_buffer,
            length=len(metrics_buffer.getvalue()),
            content_type='application/json'
        )
        
        print(f"모델 및 아티팩트 저장 완료: {bucket}/models/")
    except Exception as e:
        raise Exception(f"MinIO에 아티팩트 저장 실패: {str(e)}")


def train_model(
    minio_endpoint: str = 'minio:9000',
    minio_access_key: str = 'minio',
    minio_secret_key: str = 'minio123',
    minio_bucket: str = "raw",
    data_object: str = 'Apart Deal.csv',
    model_bucket: str = "models",
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
            save_artifacts_to_minio(
                client=client,
                bucket=model_bucket,
                model=rfc,
                encoders=encoders,
                metrics=metrics
            )
        
        return metrics
        
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        raise


# 스크립트 직접 실행 시
if __name__ == "__main__":
    train_model()
