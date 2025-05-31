# search_api.py

import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from cosin_sim_sample import OverviewSearch
from llm_question_converter import LLMQuestionConverter

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ——— 環境変数／定数 ——————————————————————————

CATALOG_DIR        = os.getenv("CATALOG_DIR",        "/static/catalog_json")
STATIC_DIR         = os.getenv("STATIC_DIR",         "/static")
SERVICE_CATALOG    = os.path.join(CATALOG_DIR,       "service_catalog.json")
EMBEDDINGS_FILE    = os.path.join(CATALOG_DIR,       "overview_embeddings.json")
USE_SAVED_EMBEDDINGS = True

# Ollama（OpenAI 互換 API）設定
LLM_URL       = os.getenv("LLM_URL",     "http://host.docker.internal:11434/v1/")
#http://localhost:11434")
LLM_API_KEY   = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL     = os.getenv("LLM_MODEL",   "llama3.3:latest")
LLM_PROMPT    = os.path.join(STATIC_DIR, "llm_service_json_prompt.txt")

logger.info(f"LLM設定: URL={LLM_URL}, MODEL={LLM_MODEL}, PROMPT={LLM_PROMPT}")

# ——— LLM ラベリングユーティリティ ———————————————————————

try:
    labeler = LLMQuestionConverter(
        llm_url         = LLM_URL,
        llm_api_key     = LLM_API_KEY,
        llm_model       = LLM_MODEL,
        llm_prompt_file = LLM_PROMPT,
    )
    logger.info("LLMQuestionConverterの初期化に成功しました")
except Exception as e:
    logger.error(f"LLMQuestionConverterの初期化に失敗しました: {str(e)}")
    raise

def label_text(text: str) -> dict[str, list[str]]:
    """
    LLMHtmlConverter.convert_html_to_json で返される JSON 配列を
    マージして { '対象者ラベル': [...], 'サービスラベル': [...] } を返す。
    """
    try:
        logger.info(f"テキストのラベル付けを開始: {text[:100]}...")
        json_list = labeler.convert_html_to_json(text)
        logger.info(f"LLMからの応答: {json_list}")
        
        if not json_list:
            logger.warning("ラベル付けの結果が空でした")
            return {}
        
        # JSON文字列を直接パース
        if isinstance(json_list, str):
            try:
                json_list = json.loads(json_list)
                logger.info(f"パース後のJSON: {json_list}")
            except json.JSONDecodeError as e:
                logger.error(f"JSONのパースに失敗しました: {str(e)}")
                return {}
        
        merged: dict[str, list[str]] = {}
        for obj in json_list:
            for k, v in obj.items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                else:
                    merged.setdefault(k, []).append(v)
        
        logger.info(f"ラベル付け結果: {merged}")
        return merged
    except Exception as e:
        logger.error(f"ラベル付け中にエラーが発生しました: {str(e)}")
        raise


# ——— サービス埋め込み＋ラベル読み込み —————————————————————

try:
    # 1) 埋め込みファイルからロード
    logger.info(f"埋め込みファイルを読み込み中: {EMBEDDINGS_FILE}")
    logger.info(f"ファイルの存在確認: {os.path.exists(EMBEDDINGS_FILE)}")
    
    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
        embeddings_data = json.load(f)
    logger.info("埋め込みファイルの読み込みに成功しました")
    logger.info(f"埋め込みデータの構造: {list(embeddings_data.keys())}")
    logger.info(f"エントリ数: {len(embeddings_data.get('entries', []))}")

    # 各サービスに埋め込みデータとラベルを追加
    services: list[dict] = []
    for i, entry in enumerate(embeddings_data.get('entries', [])):
        if isinstance(entry, dict):
            # 埋め込みデータの存在確認
            if i >= len(embeddings_data.get('embeddings', [])):
                logger.warning(f"埋め込みデータが不足しています: {entry.get('formal_name', 'unknown')}")
                continue
            
            # 埋め込みデータを追加
            entry["embedding"] = embeddings_data['embeddings'][i]
            
            # ラベルの処理
            target_labels = entry.get("target_labels", [])
            service_labels = entry.get("service_labels", [])
            
            # ラベルが文字列の場合はリストに変換
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            if isinstance(service_labels, str):
                service_labels = [service_labels]
                
            # ラベルの正規化（空白除去、重複除去）
            target_labels = [label.strip() for label in target_labels if label.strip()]
            service_labels = [label.strip() for label in service_labels if label.strip()]
            
            entry["labels"] = list(set(target_labels + service_labels))
            logger.debug(f"サービス '{entry.get('formal_name')}' のラベル: {entry['labels']}")
            services.append(entry)
    
    logger.info(f"有効なサービス数: {len(services)}")
    if services:
        logger.info(f"最初のエントリのキー: {list(services[0].keys())}")
        logger.info(f"最初のエントリのラベル: {services[0].get('labels')}")
        logger.info(f"最初のエントリの埋め込みデータの型: {type(services[0]['embedding'])}")
        logger.info(f"最初のエントリの埋め込みデータの長さ: {len(services[0]['embedding'])}")
except Exception as e:
    logger.error(f"埋め込みデータの読み込みに失敗しました: {str(e)}")
    raise


# ——— 検索用 LLM クライアント ——————————————————————————

try:
    searcher = OverviewSearch(
        service_catalog_file = SERVICE_CATALOG,
        embeddings_file      = EMBEDDINGS_FILE,
        use_saved_embeddings = USE_SAVED_EMBEDDINGS,
    )
    logger.info("OverviewSearchの初期化に成功しました")
except Exception as e:
    logger.error(f"OverviewSearchの初期化に失敗しました: {str(e)}")
    raise


def search_subset_by_embedding(question: str, subset: list[dict], top_n: int = 10):
    """
    質問文の埋め込みを取得し、subset 内の precomputed embedding
    (entry['embedding']) とコサイン類似度を計算、上位 top_n を返却。
    """
    try:
        logger.info(f"検索を開始: {question[:100]}...")
        
        # 埋め込みデータの構造を確認
        if not subset:
            logger.error("検索対象のサブセットが空です")
            return []
            
        # 最初のエントリの構造をログ出力
        logger.info(f"埋め込みデータの構造: {list(subset[0].keys())}")
        logger.info(f"サブセットのサイズ: {len(subset)}")
        
        # 質問埋め込み
        q_emb = searcher.get_embedding(question)[0]  # (d,)
        logger.info(f"質問の埋め込みベクトルの長さ: {len(q_emb)}")
        
        # サービス埋め込み行列
        svc_embs = []
        valid_entries = []
        
        for i, entry in enumerate(subset):
            if "embedding" not in entry:
                logger.warning(f"埋め込みデータが見つかりません: {entry.get('formal_name', 'unknown')}")
                if i < 3:  # 最初の3エントリの詳細を表示
                    logger.info(f"エントリの詳細: {json.dumps(entry, ensure_ascii=False, indent=2)}")
                continue
            svc_embs.append(entry["embedding"])
            valid_entries.append(entry)
            
        if not svc_embs:
            logger.error("有効な埋め込みデータが見つかりません")
            return []
            
        logger.info(f"有効な埋め込みデータ数: {len(svc_embs)}")
        logger.info(f"最初の埋め込みベクトルの長さ: {len(svc_embs[0])}")
            
        # 類似度計算
        sims = cosine_similarity([q_emb], svc_embs)[0]
        # 類似度降順でソート
        idx_sorted = np.argsort(sims)[::-1][:top_n]
        results = [(valid_entries[i], float(sims[i])) for i in idx_sorted]
        
        logger.info(f"検索結果数: {len(results)}")
        return results
    except Exception as e:
        logger.error(f"検索中にエラーが発生しました: {str(e)}")
        raise


# ——— FastAPI アプリケーション ——————————————————————————

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    question: str
    top_n: int = 10


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>行政サービス検索</title>
            <meta charset="utf-8">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .search-container {
                    margin: 20px 0;
                }
                input[type="text"] {
                    width: 70%;
                    padding: 10px;
                    font-size: 16px;
                }
                button {
                    padding: 10px 20px;
                    font-size: 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                .result-item {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .result-item h3 {
                    margin-top: 0;
                }
                .score {
                    color: #666;
                    font-size: 0.9em;
                }
                .labels-container {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                }
                .label {
                    display: inline-block;
                    padding: 2px 8px;
                    margin: 2px;
                    border-radius: 12px;
                    font-size: 0.9em;
                }
                .target-label {
                    background-color: #e3f2fd;
                    color: #1976d2;
                }
                .service-label {
                    background-color: #f3e5f5;
                    color: #7b1fa2;
                }
                .result-labels {
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h1>行政サービス検索</h1>
            <div class="search-container">
                <input type="text" id="searchInput" placeholder="検索したい内容を入力してください">
                <button onclick="search()">検索</button>
            </div>
            <div id="userLabels" class="labels-container"></div>
            <div id="results"></div>

            <script>
                async function search() {
                    const question = document.getElementById('searchInput').value;
                    if (!question) return;

                    try {
                        const response = await fetch('/search', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                question: question,
                                top_n: 10
                            })
                        });

                        const data = await response.json();
                        const resultsDiv = document.getElementById('results');
                        const userLabelsDiv = document.getElementById('userLabels');
                        resultsDiv.innerHTML = '';
                        userLabelsDiv.innerHTML = '';

                        // ユーザーラベルの表示
                        if (data.user_labels) {
                            const targetLabels = data.user_labels["対象者ラベル"] || [];
                            const serviceLabels = data.user_labels["サービスラベル"] || [];
                            
                            if (targetLabels.length > 0 || serviceLabels.length > 0) {
                                let labelsHtml = '';
                                targetLabels.forEach(label => {
                                    labelsHtml += `<span class="label target-label">${label}</span>`;
                                });
                                serviceLabels.forEach(label => {
                                    labelsHtml += `<span class="label service-label">${label}</span>`;
                                });
                                userLabelsDiv.innerHTML = labelsHtml;
                            }
                        }

                        // 検索結果の表示
                        data.results.forEach(result => {
                            const resultItem = document.createElement('div');
                            resultItem.className = 'result-item';
                            
                            // マッチング情報からラベルを表示
                            let labelsHtml = '';
                            const matchInfo = result.match_info;
                            if (matchInfo) {
                                if (matchInfo.matched_target_labels) {
                                    matchInfo.matched_target_labels.forEach(label => {
                                        labelsHtml += `<span class="label target-label">${label}</span>`;
                                    });
                                }
                                if (matchInfo.matched_service_labels) {
                                    matchInfo.matched_service_labels.forEach(label => {
                                        labelsHtml += `<span class="label service-label">${label}</span>`;
                                    });
                                }
                            }
                            
                            resultItem.innerHTML = `
                                <h3>${result.formal_name}</h3>
                                <p>${result.overview}</p>
                                <div class="result-labels">${labelsHtml}</div>
                                <p><a href="${result.url}" target="_blank">詳細を見る</a></p>
                                <p class="score">類似度: ${(result.score * 100).toFixed(1)}%</p>
                            `;
                            resultsDiv.appendChild(resultItem);
                        });
                    } catch (error) {
                        console.error('Error:', error);
                        alert('検索中にエラーが発生しました。');
                    }
                }

                // Enterキーでも検索を実行
                document.getElementById('searchInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        search();
                    }
                });
            </script>
        </body>
    </html>
    """


@app.post("/search")
async def search(req: SearchRequest):
    try:
        user_q = req.question
        logger.info(f"検索リクエスト受信: {user_q}")

        # 1) ユーザークエリをラベル付け
        user_labels = label_text(user_q)
        user_target_labels = set(user_labels.get("対象者ラベル", []))
        user_service_labels = set(user_labels.get("サービスラベル", []))
        
        logger.info(f"検出された対象者ラベル: {user_target_labels}")
        logger.info(f"検出されたサービスラベル: {user_service_labels}")

        # 2) サービスをラベルでフィルター
        filtered = []
        for svc in services:
            # サービスラベルの取得と正規化
            svc_service_labels = set(label.strip() for label in svc.get("service_labels", []))
            svc_target_labels = set(label.strip() for label in svc.get("target_labels", []))
            
            logger.debug(f"サービス '{svc.get('formal_name')}' のサービスラベル: {svc_service_labels}")
            
            # サービスラベルでマッチするか確認（部分一致）
            service_match = bool(user_service_labels.intersection(svc_service_labels))
            
            if service_match:
                # 対象者ラベルの一致も確認
                target_match = bool(user_target_labels.intersection(svc_target_labels))
                
                # マッチング情報をサービスに追加
                matched_service_labels = list(user_service_labels.intersection(svc_service_labels))
                matched_target_labels = list(user_target_labels.intersection(svc_target_labels)) if target_match else []
                
                svc["match_info"] = {
                    "service_match": True,
                    "target_match": target_match,
                    "matched_service_labels": matched_service_labels,
                    "matched_target_labels": matched_target_labels,
                    "all_service_labels": list(svc_service_labels),  # デバッグ用に全ラベルも記録
                    "all_target_labels": list(svc_target_labels)     # デバッグ用に全ラベルも記録
                }
                
                # スコアの設定（対象者ラベルが一致する場合は1.0、そうでない場合は0.5）
                svc["score"] = 1.0 if target_match else 0.5
                
                filtered.append(svc)
                logger.info(f"サービス '{svc.get('formal_name')}' がマッチしました")
                logger.info(f"  - サービスラベルの一致: {matched_service_labels}")
                if target_match:
                    logger.info(f"  - 対象者ラベルの一致: {matched_target_labels}")

        if not filtered:
            logger.warning("ラベルによるフィルタリング結果が空のため、全件検索にフォールバック")
            filtered = services
            # 全件検索の場合はスコアを0.1に設定
            for svc in filtered:
                svc["score"] = 0.1
                svc["match_info"] = {
                    "service_match": False,
                    "target_match": False,
                    "matched_service_labels": [],
                    "matched_target_labels": [],
                    "all_service_labels": list(svc.get("service_labels", [])),
                    "all_target_labels": list(svc.get("target_labels", []))
                }

        logger.info(f"フィルタリング後のサービス数: {len(filtered)}")
        if filtered:
            logger.info(f"最初のサービス: {filtered[0].get('formal_name')}")
            logger.info(f"最初のサービスの対象者ラベル: {filtered[0].get('target_labels')}")
            logger.info(f"最初のサービスのサービスラベル: {filtered[0].get('service_labels')}")

        # 3) 埋め込みベクトルによる類似度評価
        embedding_results = search_subset_by_embedding(user_q, filtered, top_n=10)
        
        # 4) レスポンス整形
        response = {
            "question":    user_q,
            "user_labels": {
                "対象者ラベル": list(user_target_labels),
                "サービスラベル": list(user_service_labels)
            },
            "results": [
                {
                    "formal_name": result[0].get("formal_name"),
                    "overview":    result[0].get("overview"),
                    "url":         result[0].get("url"),
                    "score":       result[1],  # 埋め込みベクトルの類似度スコア
                    "match_info":  result[0].get("match_info", {})
                }
                for result in embedding_results
            ]
        }
        logger.info(f"検索完了: {len(response['results'])}件の結果を返却")
        return response

    except Exception as e:
        logger.error(f"検索処理中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

