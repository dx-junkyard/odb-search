from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from cosin_sim_sample import OverviewSearch
import os

# FastAPIアプリケーションの初期化
app = FastAPI()

# "static"ディレクトリをマウント
app.mount("/static", StaticFiles(directory="/static"), name="static")

# ディレクトリマウントを利用したサービスカタログと埋め込みデータのファイルパス
CATALOG_DIR = os.getenv("CATALOG_DIR", "./catalog_json")
SERVICE_CATALOG_FILE = os.path.join(CATALOG_DIR, "service_catalog.json")
EMBEDDINGS_FILE = os.path.join(CATALOG_DIR, "overview_embeddings.json")

# OverviewSearchインスタンスを初期化
overview_search = OverviewSearch(
    service_catalog_file=SERVICE_CATALOG_FILE,
    embeddings_file=EMBEDDINGS_FILE,
    use_saved_embeddings=True
)

# リクエストボディのモデル定義
class SearchRequest(BaseModel):
    question: str
    top_n: int = 3

@app.post("/search")
async def search(request: SearchRequest):
    try:
        # 質問とトップNを取得
        question = request.question
        top_n = request.top_n

        if not question:
            raise HTTPException(status_code=400, detail="質問が提供されていません。")

        # 類似する概要を検索
        results = overview_search.search_top_n_similar_overviews(question, top_n=top_n)

        # 結果を整形して返す
        formatted_results = [
            {
                "formal_name": result["formal_name"],
                "overview": result["overview"],
                "url": result["url"],
                "score": score
            }
            for result, score in results
        ]

        return {"question": question, "results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

