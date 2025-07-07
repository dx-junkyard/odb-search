"""Load catalog, filter by labels, and rank by BERT cosine similarity."""
import json, os, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "catalog_json", "service_catalog.json")
EMBED_PATH   = os.path.join(os.path.dirname(__file__), "..", "static", "catalog_json", "overview_embeddings.json")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    _catalog_raw = json.load(f)
CATALOG_DF = pd.DataFrame(_catalog_raw)

_embed_raw: list | None = None
if os.path.exists(EMBED_PATH) and os.path.getsize(EMBED_PATH) > 0:
    try:
        with open(EMBED_PATH, "r", encoding="utf-8") as f:
            _embed_raw = json.load(f).get("embeddings")
    except Exception:
        _embed_raw = None

if _embed_raw:
    EMBED_MATRIX = np.array(_embed_raw, dtype=np.float32)
else:
    # fail-safe when embedding file is missing or invalid
    EMBED_MATRIX = np.empty((0, 768), dtype=np.float32)


def apply_label_filter(df: pd.DataFrame, tgt: list[str], svc: list[str]):
    """
    ラベルによるフィルタリング
    1. サービスラベルと対象者ラベルそれぞれで一致する要素がある
    2. サービスラベルで一致する要素がある
    3. 対象者ラベルで一致する要素がある
    4. サービスラベルと対象者ラベルともに一致する要素がない場合は全件取得
    """
    if not tgt and not svc:
        return df
    
    # NaN値を安全に処理する関数
    def safe_any_match(labels, target_list):
        try:
            # None または NaN の場合
            if labels is None:
                return False
            
            # NumPy配列やPandas Seriesの場合
            if hasattr(labels, '__iter__') and not isinstance(labels, (str, list)):
                # NaN値を含むかチェック
                if hasattr(labels, 'any') and labels.any():
                    # 配列にNaNが含まれている場合
                    if hasattr(labels, 'isna') and labels.isna().any():
                        return False
                    # 配列をリストに変換
                    labels = list(labels)
                else:
                    # 空の配列の場合
                    return False
            
            # リストでない場合はFalse
            if not isinstance(labels, list):
                return False
            
            # 空のリストの場合
            if len(labels) == 0:
                return False
            
            # ラベルマッチング
            return any(l in labels for l in target_list)
            
        except Exception:
            # エラーが発生した場合はFalseを返す
            return False
    
    # 各条件のマスクを作成
    tgt_match = df["対象者ラベル"].apply(lambda x: safe_any_match(x, tgt)) if tgt else pd.Series([False] * len(df))
    svc_match = df["サービスラベル"].apply(lambda x: safe_any_match(x, svc)) if svc else pd.Series([False] * len(df))
    
    # 条件1: サービスラベルと対象者ラベルそれぞれで一致する要素がある
    condition1 = tgt_match & svc_match
    
    # 条件2: サービスラベルで一致する要素がある
    condition2 = svc_match
    
    # 条件3: 対象者ラベルで一致する要素がある
    condition3 = tgt_match
    
    # いずれかの条件に一致するものを返す
    final_mask = condition1 | condition2 | condition3
    
    # 条件4: 一致する要素がない場合は全件取得
    if not final_mask.any():
        return df
    
    return df[final_mask]


def rank_by_similarity(filtered: pd.DataFrame, query_vec: np.ndarray):
    if filtered.empty:
        return filtered
    
    # 有効なインデックスのみを取得（EMBED_MATRIXの範囲内）
    valid_indices = filtered.index[filtered.index < len(EMBED_MATRIX)]
    
    if len(valid_indices) == 0:
        # 有効なインデックスがない場合は空のDataFrameを返す
        return filtered.iloc[:0]
    
    # 有効なインデックスのみでフィルタリング
    filtered_valid = filtered.loc[valid_indices]
    
    # 類似度計算
    sims = cosine_similarity([query_vec], EMBED_MATRIX[valid_indices])[0]
    out = filtered_valid.copy()
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(10)


class CatalogSearchEngine:
    """Search and rank services within the catalog."""

    def __init__(self, catalog_df: pd.DataFrame = CATALOG_DF, embed_matrix: np.ndarray = EMBED_MATRIX):
        self.catalog_df = catalog_df
        self.embed_matrix = embed_matrix

    def filter_by_labels(self, target_labels: list[str], service_labels: list[str]):
        return apply_label_filter(self.catalog_df, target_labels, service_labels)

    def rank(self, filtered_df: pd.DataFrame, query_vec: np.ndarray, top_n: int = 10):
        if filtered_df.empty:
            return filtered_df

        valid_indices = filtered_df.index[filtered_df.index < len(self.embed_matrix)]
        if len(valid_indices) == 0:
            return filtered_df.iloc[:0]

        filtered_valid = filtered_df.loc[valid_indices]
        sims = cosine_similarity([query_vec], self.embed_matrix[valid_indices])[0]
        out = filtered_valid.copy()
        out["similarity"] = sims
        return out.sort_values("similarity", ascending=False).head(top_n)
