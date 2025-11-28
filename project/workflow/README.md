# DataOps, MLOps workflow management 

- 아래 순서로 "docker compose up" 진행

## How to start 
- docker compose -f compose.airflow.yaml up airflow-init
- docker compose -f compose.airflow.yaml up -d

## How to make .env
- If there is no .env file, create .env in the workflow path.
- command: $ echo -e "AIRFLOW_UID=$(id -u)" > .env
