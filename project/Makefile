copy-makefile:
	rsync -avh ./Makefile /home/pcnrnd/population/Makefile

copy-dags:
	rsync -avh ./workflow/dags/ /home/pcnrnd/population/workflow/dags/


up-airflow-init:
	cd workflow && docker compose -f compose.airflow.yaml up airflow-init

up-airflow:
	cd workflow && docker compose -f compose.airflow.yaml up -d

down-airflow:
	cd workflow && docker compose -f compose.airflow.yaml down --volumes --rmi all

up-mlflow:
	cd mlflow && docker compose -f compose.mlflow.yml up -d

down-mlflow:
	cd mlflow && docker compose -f compose.mlflow.yml down --volumes --rmi all

up-metadata:
	cd metadata && docker compose -f compose.postgres.yaml up -d

down-metadata:
	cd metadata && docker compose -f compose.postgres.yaml down --volumes --rmi all

up-all:
	make up-airflow-init
	make up-airflow
	make up-mlflow
	make up-metadata
