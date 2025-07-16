FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8506

CMD ["streamlit", "run", "chart_type.py", "--server.port", "8506", "--server.address", "0.0.0.0"]