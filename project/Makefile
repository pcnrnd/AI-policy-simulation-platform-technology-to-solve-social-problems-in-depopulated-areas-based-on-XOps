# 디렉토리 변수 정의
WORKFLOW_DIR := workflow
MLFLOW_DIR := mlflow
METADATA_DIR := metadata

# Airflow
up-airflow-init:
	docker compose -C $(WORKFLOW_DIR) -f compose.airflow.yaml up airflow-init

up-airflow:
	docker compose -C $(WORKFLOW_DIR) -f compose.airflow.yaml up -d

down-airflow:
	docker compose -C $(WORKFLOW_DIR) -f compose.airflow.yaml down --volumes --rmi all

# MLflow
up-mlflow:
	docker compose -C $(MLFLOW_DIR) -f compose.mlflow.yml up -d

down-mlflow:
	docker compose -C $(MLFLOW_DIR) -f compose.mlflow.yml down --volumes --rmi all

# Metadata
up-metadata:
	docker compose -C $(METADATA_DIR) -f compose.postgres.yaml up -d

down-metadata:
	docker compose -C $(METADATA_DIR) -f compose.postgres.yaml down --volumes --rmi all

# 전체 실행
up-all:
	@echo "Starting all services..."
	@$(MAKE) up-airflow-init
	@$(MAKE) up-airflow
	@$(MAKE) up-mlflow
	@$(MAKE) up-metadata
	@echo "All services started successfully"

# 전체 종료
down-all:
	@echo "Stopping all services..."
	@$(MAKE) down-airflow || true
	@$(MAKE) down-mlflow || true
	@$(MAKE) down-metadata || true
	@echo "All services stopped"