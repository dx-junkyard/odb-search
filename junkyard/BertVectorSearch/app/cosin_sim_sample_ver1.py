import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# BERTモデルの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# JSONデータの読み込み
with open('service_catalog_765.json', 'r', encoding='utf-8') as f:
    service_catalog = json.load(f)

# 質問文をベクトル化
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()


# 全ての「概要」をベクトル化
def get_overview_embeddings(service_catalog):
    overview_embeddings = []
    overviews = []
    for service in service_catalog:
        overview_data = service.get("概要", {}).get("items", [])
        if isinstance(overview_data, list):
            overview = " ".join(overview_data)
        elif isinstance(overview_data, str):
            overview = overview_data
        else:
            continue  # 何らかの不正な形式の場合はスキップ

        if overview:
            overview_embeddings.append(get_embedding(overview))
            overviews.append(overview)
    return np.vstack(overview_embeddings), overviews

# 全ての「概要」をベクトル化
def get_overview_embeddings_old(service_catalog):
    overview_embeddings = []
    overviews = []
    for service in service_catalog:
        overview = " ".join(service.get("概要", {}).get("items", []))
        if overview:
            overview_embeddings.append(get_embedding(overview))
            overviews.append(overview)
    return np.vstack(overview_embeddings), overviews

# コサイン類似度を計算して最も類似した「概要」を取得
def search_most_similar_overview(question, service_catalog):
    question_embedding = get_embedding(question)
    overview_embeddings, overviews = get_overview_embeddings(service_catalog)
    
    # コサイン類似度の計算
    similarities = cosine_similarity(question_embedding, overview_embeddings)
    most_similar_index = np.argmax(similarities)

    return overviews[most_similar_index]

# 質問文
question = "夫婦共働きで、子育て中の母親です。休みの日などで子どもと遊べる場所、子どもを預かってくれるサービスなど教えてほしいです。"
print("質問文", question)

# 検索実行
most_similar_overview = search_most_similar_overview(question, service_catalog)
print("最も類似した概要: ", most_similar_overview)

