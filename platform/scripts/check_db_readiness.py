"""Read-only DB readiness gate for the separated population-db stack.

Run inside the existing xops-service container (it already has the DSN env and DB drivers):
  docker exec -i platform-xops-service-1 python - < platform/scripts/check_db_readiness.py

Checks only: SELECT 1, that the public schema still has tables, and Mongo ping + collections.
An empty database means a fresh volume or re-run init, so it fails instead of passing.
Never prints DSNs, credentials or row contents; never imports the app or calls init_db.
"""
import json
import os
import sys

TIMEOUT_SECONDS = 5
results = []


def record(name, ok, **detail):
    results.append({"check": name, "result": "PASS" if ok else "FAIL", **detail})


def check_postgres(name, env_key):
    dsn = os.environ.get(env_key)
    if not dsn:
        record(name, False, reason=f"{env_key} 미설정")
        return
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=TIMEOUT_SECONDS) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
                cur.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
                tables = cur.fetchone()[0]
        record(name, tables > 0, public_tables=tables)
    except Exception as exc:  # 접속·인증·드라이버 실패를 모두 FAIL로 기록한다
        record(name, False, reason=type(exc).__name__)


def check_mongo():
    uri = os.environ.get("XOPS_MONGO_URI")
    if not uri:
        record("mongo", False, reason="XOPS_MONGO_URI 미설정")
        return
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=TIMEOUT_SECONDS * 1000)
        try:
            client.admin.command("ping")
            collections = len(client.get_default_database().list_collection_names())
        finally:
            client.close()
        record("mongo", collections > 0, collections=collections)
    except Exception as exc:
        record("mongo", False, reason=type(exc).__name__)


check_postgres("pg", "XOPS_PG_DSN")
check_postgres("timescale", "XOPS_TIMESCALE_DSN")
check_mongo()
for row in results:
    print(json.dumps(row, ensure_ascii=False))
sys.exit(0 if all(row["result"] == "PASS" for row in results) else 1)
