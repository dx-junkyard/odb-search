FROM python:3.9-slim

WORKDIR /app

# 必要なディレクトリを作成
RUN mkdir -p /static

# アプリケーションコードと静的ファイルをコピー
COPY ./app /app
COPY ./static /static

# 依存関係をインストール
RUN pip install --no-cache-dir -r /app/requirements.txt

# 環境変数を設定
ENV PYTHONUNBUFFERED=1
ENV CATALOG_DIR=/static/catalog_json

# アプリケーションを起動
CMD ["uvicorn", "search_api:app", "--host", "0.0.0.0", "--port", "8000"]

