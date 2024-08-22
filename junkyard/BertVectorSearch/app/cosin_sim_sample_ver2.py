import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class OverviewSearch:
    def __init__(self, service_catalog_file, embeddings_file=None, use_saved_embeddings=False):
        """
        :param service_catalog_file: JSONファイルのパス (service_catalog.json)
        :param embeddings_file: ベクトルデータを保存/読み込むためのJSONファイルのパス (overview_embeddings.json)
        :param use_saved_embeddings: Trueにすると既存のembeddings_fileを使用し、Falseにするとservice_catalog_fileを元に生成する
        """
        self.service_catalog_file = service_catalog_file
        self.embeddings_file = embeddings_file
        self.use_saved_embeddings = use_saved_embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        if self.use_saved_embeddings and self.embeddings_file and os.path.exists(self.embeddings_file):
            self.overview_embeddings, self.entries = self.load_embeddings_from_file(self.embeddings_file)
        else:
            self.service_catalog = self.load_json_data(self.service_catalog_file)
            self.overview_embeddings, self.entries = self.get_overview_embeddings(self.service_catalog)
            if self.embeddings_file:
                self.save_embeddings_to_file(self.overview_embeddings, self.entries, self.embeddings_file)
    
    def load_json_data(self, file_path):
        """JSONファイルを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_embeddings_to_file(self, embeddings, entries, output_file):
        """ベクトルデータをJSONファイルに保存する"""
        data = {
            'embeddings': embeddings.tolist(),
            'entries': entries
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_embeddings_from_file(self, input_file):
        """JSONファイルからベクトルデータを読み込む"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return np.array(data['embeddings']), data['entries']

    def get_embedding(self, text):
        """テキストをベクトル化する"""
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.detach().numpy()

    def get_overview_embeddings(self, service_catalog):
        """全ての概要をベクトル化する"""
        overview_embeddings = []
        entries = []
        
        for service in service_catalog:
            overview_data = service.get("概要", {}).get("items", [])
            if isinstance(overview_data, list):
                overview = " ".join(overview_data)
            elif isinstance(overview_data, str):
                overview = overview_data
            else:
                continue  # 不正な形式はスキップ

            if overview:
                embedding = self.get_embedding(overview)
                overview_embeddings.append(embedding)
                
                entry = {
                    'overview': overview,
                    'formal_name': service.get("正式名称", {}).get("items", ["N/A"])[0],
                    'url': service.get("URL", {}).get("items", "N/A")
                }
                entries.append(entry)

        return np.vstack(overview_embeddings), entries

    def search_most_similar_overview(self, question):
        """質問に最も類似した概要を返す"""
        question_embedding = self.get_embedding(question)
        similarities = cosine_similarity(question_embedding, self.overview_embeddings)
        most_similar_index = np.argmax(similarities)
        return self.entries[most_similar_index]

# 使用例
service_catalog_file = "service_catalog.json"
embeddings_file = "overview_embeddings.json"
question = "子育て中の母親です。休みの日などで子どもと遊べる場所、子どもを預かってくれるサービスなど教えてほしいです。"

# クラスの初期化
overview_search = OverviewSearch(service_catalog_file, embeddings_file, use_saved_embeddings=True)

# 質問に対する最も類似した結果を取得
most_similar_entry = overview_search.search_most_similar_overview(question)

# 検索結果の表示
print("最も類似した概要: ", most_similar_entry['overview'])
print("正式名称: ", most_similar_entry['formal_name'])
print("URL: ", most_similar_entry['url'])

