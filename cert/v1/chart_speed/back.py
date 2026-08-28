from fastapi import FastAPI, Request
import duckdb
import json

app = FastAPI()
conn = duckdb.connect('./data/database.db')

@app.post('/data')
def get_data(table: Request):

    df = conn.execute(f"SELECT * FROM {table}").df()
    
    dumps_data = json.dumps(df)
    json_data = json.loads(dumps_data)
    # DataFrame을 JSON으로 변환하여 반환
    return json_data