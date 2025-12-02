import mlflow   
import os
import requests
import pandas as pd
from minio import Minio
from io import BytesIO

# ======================
# LightGBM 학습 (불균형 보정 + 조기종료) - MLflow 연동
# ======================
# 테스트용 데이터 임의 생성 (입력)
import numpy as np
import pandas as pd
import os
import pickle
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import socket


mlflow_server_ip = socket.gethostbyname("mlflow-server")
mlflow.set_tracking_uri(f"http://{mlflow_server_ip}:5000")
mlflow.set_experiment("test_classifier")

# MinIO 클라이언트 초기화
client = Minio(
    "mlflow-minio:9000",  # MinIO 서버 주소
    access_key="minio",
    secret_key="minio123",
    secure=False  # HTTPS를 사용하지 않는 경우 (프로덕션 환경에서는 True로 설정)
)

# 모든 년도 파일 리스트
bucket_name = "prepro"
years = ['2020', '2021', '2022', '2023', '2024', '2025']
dfs = []

for year in years:
    object_name = f"restaurant_{year}.csv"
    response = client.get_object(bucket_name, object_name)
    data_stream = BytesIO(response.read())
    df_temp = pd.read_csv(data_stream)
    dfs.append(df_temp)
    response.close()
    response.release_conn()

# 모든 DataFrame 합치기
df = pd.concat(dfs, ignore_index=True)



le = LabelEncoder()
label_data = le.fit_transform(df['영업상태명'])

df['label'] = label_data

# 주소지 컬럼 인코딩
import re

# 번지 숫자 추출 함수
def extract_lot_number(lot_str):
    """번지에서 숫자를 추출합니다 (예: '3', '11', '169-3' -> 3, 11, 169)"""
    if pd.isna(lot_str) or lot_str == '' or lot_str == 'nan':
        return 0
    # 첫 번째 숫자 추출
    match = re.search(r'\d+', str(lot_str))
    return int(match.group()) if match else 0

# 번지 컬럼 숫자로 변환
df['번지_숫자'] = df['번지'].apply(extract_lot_number)

# 시도, 시군구, 읍면동 LabelEncoder 인코딩
cols = ['시도', '시군구', '읍면동', '구분']
address_encoders = {}

for col in cols:
    if col in df.columns:
        le = LabelEncoder()
        # NaN 값을 '기타'로 채우기
        df[col] = df[col].fillna('기타')
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        address_encoders[col] = le

# 사용할 feature 선정
unused_cols = ['개방서비스명', '인허가일자', '폐업일자', '영업상태명', '소재지', 
               'label', '시도', '시군구', '읍면동', '번지', '도로명', 
               '시설명', '남성종사자수', '여성종사자수']  # 시설명, 종사자수도 제외
features = [col for col in df.columns if col not in unused_cols]

X = df[features]
y = df['label']

import os

# MinIO/S3 접근을 위한 AWS credentials 설정
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'  # .env.train 파일의 값으로 변경
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'  # .env.train 파일의 값으로 변경
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://mlflow-minio:9000'
os.environ['AWS_ENDPOINT_URL'] = 'http://mlflow-minio:9000'




cat_features = []
for col in X_train.columns:
    if "encoded" in col or X_train[col].dtype == "object":
        cat_features.append(col)


with mlflow.start_run():
    # 정수 컬럼 경고 해결: 전부 float64로 변환
    # 문자열 컬럼이 있다면 float로 변환 가능한 컬럼만 선택
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train = X_train[numeric_cols].astype('float64')
    X_test = X_test[numeric_cols].astype('float64')
    
    # cat_features도 업데이트 (X_train에 존재하는 컬럼만)
    cat_features_updated = [col for col in cat_features if col in numeric_cols]
    
    # 파라미터 로깅
    mlflow.log_params({
        "n_estimators": 1500,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "test_size": 0.2,
        "random_state": 123,
        "categorical_features": cat_features_updated
    })
    
    lgbm = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # 학습 메트릭을 수집하기 위한 callback
    evals_result = {}
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=True),
            lgb.record_evaluation(evals_result)
        ],
        categorical_feature=cat_features_updated if cat_features_updated else "auto"
    )
    
    # 학습 과정 메트릭 로깅 (시간 경과 시각화용)
    for metric_name, metric_values in evals_result.get('valid_0', {}).items():
        for epoch, value in enumerate(metric_values):
            mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
    
    # 모델 저장 (로컬 파일)
    model_filename = f"lgbm_classifier_run_{mlflow.active_run().info.run_id}.pkl"  # 동적 이름
    with open(os.path.join(ARTIFACT_DIR, model_filename), "wb") as f:
        pickle.dump(lgbm, f)
    
    # 평가 메트릭을 MLflow로 로깅
    proba = lgbm.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    
    roc = roc_auc_score(y_test, proba)
    precision = average_precision_score(y_test, proba)
    
    mlflow.log_metrics({
        "roc": roc,
        "precision": precision,
        "f1_score": f1_score(y_test, pred_default),
        "accuracy": (y_test == pred_default).mean(),
        "best_iteration": lgbm.best_iteration_,
        "num_features": len(X_train.columns)
    })
    
    # 모델 로깅 (MLflow에 모델도 저장, 개선된 방식: signature와 input_example 포함)
    from mlflow.models import infer_signature

    # 1. 모델이 예상하는 입력 예제 생성 (float64 유지)
    input_example = X_train.iloc[:5]

    # 2. 모델의 예측 결과로 시그니처 추론
    signature = mlflow.models.infer_signature(
        X_train.iloc[:5],
        lgbm.predict(X_train.iloc[:5])
    )

    # 3. 시그니처와 예제를 포함하여 모델 저장
    mlflow.lightgbm.log_model(
        lgbm, 
        name="lgbm_classifier",  # ← 여기서 모델 이름 변경 가능
        signature=signature,
        input_example=input_example
    )
    
    print(f"\n[MLflow] 실험 저장 완료!")