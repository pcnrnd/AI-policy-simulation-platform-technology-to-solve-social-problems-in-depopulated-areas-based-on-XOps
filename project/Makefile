up-airflow-init:
	docker compose -f ./workflow/compose.airflow.yaml up airflow-init

up-airflow:
	docker compose -f ./workflow/compose.airflow.yaml up -d

down-airflow:
	docker compose -f ./workflow/compose.airflow.yaml down --volumes --rmi all

up-mlflow:
	docker compose -f ./mlflow/compose.mlflow.yml up -d

down-mlflow:
	docker compose -f ./mlflow/compose.mlflow.yml down --volumes --rmi all

up-metadata:
	docker compose -f ./metadata/compose.postgres.yaml up -d

down-metadata:
	docker compose -f ./metadata/compose.postgres.yaml down --volumes --rmi all

up-all:
	make up-airflow-init
	make up-airflow
	make up-mlflow
	make up-metadata

down-all:
	docker compose -f ./workflow/compose.airflow.yaml down --volumes --rmi all
	docker compose -f ./mlflow/compose.mlflow.yml down --volumes --rmi all
	docker compose -f ./metadata/compose.postgres.yaml down --volumes --rmi all