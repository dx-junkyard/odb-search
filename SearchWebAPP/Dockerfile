FROM python:3.11-slim
WORKDIR /app

# System deps for transformers & torch wheels
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY app/ app/
COPY static/ static/
COPY app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]

