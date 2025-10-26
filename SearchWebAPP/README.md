# SearchWebAPP

自治体サービス検索システムの Streamlit アプリケーションです。自治体が提供するサービスカタログを検索し、OpenAI API を活用してユーザーの質問意図に沿った情報を提示します。

## 構成

- `app/`
  - `main.py`: Streamlit エントリーポイント。
  - `catalog_utils.py`: サービスカタログの検索・ランキング処理。
  - `conversation_graph.py`: LangGraph を使った対話制御ロジック。
  - `llm_utils.py`: OpenAI Chat Completions API を用いた意図推定・推薦ロジック。
  - `embed_utils.py`: OpenAI Embeddings API によるベクトル生成。
  - `requirements.txt`: アプリで必要となる Python 依存関係。
- `static/`
  - `catalog_json/`: サービスカタログの JSON データ。
  - `llm_service_json_prompt.txt`, `llm_service_select_prompt.txt`: LLM 向けプロンプトテンプレート。
- `Dockerfile`, `docker-compose.yaml`: コンテナ開発用の設定。

## 必要条件

- Python 3.10 以降
- OpenAI API キー (`OPENAI_API_KEY`)
- ネットワーク接続（OpenAI API へアクセスするため）

任意で以下の環境変数を設定できます。

- `LLM_MODEL`: Chat Completions 用モデル名。既定値は `gpt-4o-mini`。
- `OPENAI_EMBEDDING_MODEL`: Embeddings 用モデル名。既定値は `text-embedding-ada-002`。

`.env` ファイルをプロジェクト直下（`SearchWebAPP/.env`）に作成すると、Streamlit 実行時や Docker コンテナ内で自動的に読み込まれます。

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

## セットアップと実行

### 1. ローカルで実行

1. 依存関係をインストールします。
   ```bash
   cd SearchWebAPP
   python -m venv .venv
   source .venv/bin/activate  # Windows の場合は .venv\Scripts\activate
   pip install -r app/requirements.txt
   ```
2. `.env` を準備し、OpenAI API キーなどを設定します。
3. Streamlit でアプリを起動します。
   ```bash
   streamlit run app/main.py
   ```
4. ブラウザで <http://localhost:8501> にアクセスします。

### 2. Docker / Docker Compose を利用

Docker を用いるとローカル環境を汚さずに開発できます。

1. `.env` を `SearchWebAPP/.env` に配置します。
2. Docker イメージをビルドし、コンテナを起動します。
   ```bash
   cd SearchWebAPP
   docker compose up --build
   ```
3. ブラウザで <http://localhost:8501> を開きます。

開発中にホットリロードを有効にするため、`docker-compose.yaml` ではプロジェクトディレクトリを `/app` にマウントしています。

## カタログデータの更新

サービスカタログの JSON は `static/catalog_json/` 以下に格納されています。データを差し替える場合は同ディレクトリにあるファイルを更新し、再度アプリを起動してください。

## ライセンス

このリポジトリのルートにある `LICENSE` を参照してください。
