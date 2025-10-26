# app/llm_utils.py
import os
import json
from typing import List, Dict, Any

from openai import OpenAI
import logging

# モデルは .env の LLM_MODEL で上書き可能（既定: gpt-4o-mini）
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)
client = OpenAI()


# --------------------------------------------------------------------
# ラベル付け：ユーザー質問から 対象者ラベル / サービスラベル を推定
# 返り値: {"target_labels": [...], "service_labels": [...]}
# --------------------------------------------------------------------
def label_question(question: str) -> Dict[str, List[str]]:
    system_prompt = (
        "あなたは自治体サービス検索のラベリング係です。"
        "入力文から『対象者ラベル』と『サービスラベル』を推定し、必ずJSONで返してください。"
        "キーは target_labels と service_labels の2つです。"
        "候補は以下のリストのみから選んでください。"
        "\n\n"
        "【対象者ラベル】\n"
        "  乳幼児（0～2歳）, 未就学児（3歳〜小学校入学前）, 小学生, 中学生, 高校生, 大学生, "
        "  保護者, 社会人, 高齢者, 障がい者, 事業者, 男性, 女性, どなたでも利用・参加可能, その他（該当が不明な場合）\n\n"
        "【サービスラベル】\n"
        "  補助金・助成金, 住まい・住宅支援, ペット・動物愛護, 水道・上下水道, 公園・緑地・レクリエーション, "
        "  意見・要望・苦情受付, 健康・医療, 福祉・介護, 子育て・教育, 雇用・就労支援, 市民生活・手続き, 防災・災害対応, "
        "  環境・ごみ・リサイクル, まちづくり・都市整備, 産業・事業者支援, 文化・スポーツ, 交通・移動支援, 移住・定住促進, "
        "  男女共同参画・人権・相談, 行政運営・計画・評価, 選挙・政治参加, デジタル・IT関連, 消費生活・トラブル対応, その他\n"
        "\n必ずJSONのみを出力してください。"
    )
    user_prompt = f"入力文: {question}"

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},  # JSON を強制
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        return {
            "target_labels": data.get("target_labels", []) or [],
            "service_labels": data.get("service_labels", []) or [],
        }
    except Exception as e:
        logger.warning("label_question JSON parse failed: %s | raw=%s", e, content[:300].replace("\n", " "))
        return {"target_labels": [], "service_labels": []}


# --------------------------------------------------------------------
# 候補サービスから最大3件を選ぶセレクタ
# candidates: [{"title": str, "url": str}, ...]
# --------------------------------------------------------------------
class ServiceSelector:
    def __init__(self, model: str = DEFAULT_MODEL, top_k: int = 3):
        self.model = model
        self.top_k = top_k

    def recommend(self, query: str, candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prompt = (
            "あなたは自治体サービスの案内係です。ユーザー質問に最も関連する候補を最大3件、"
            "JSON で返してください。出力形式:\n"
            '{"recommendations":[{"title":"...","url":"..."}, ...]}\n\n'
            f"ユーザー質問: {query}\n\n候補:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},  # JSON を強制
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
            recs = parsed.get("recommendations", []) or []
            normed = []
            for r in recs:
                t = str(r.get("title", "")).strip()
                u = r.get("url", "")
                if t:
                    normed.append({"title": t, "url": u if isinstance(u, str) else u})
            return normed[: self.top_k]
        except Exception as e:
            logger.warning("ServiceSelector JSON parse failed: %s | raw=%s", e, content[:300].replace("\n", " "))
            return []


# --------------------------------------------------------------------
# 検索用「正規化1文」ビルダー
# 返り値:
#   {
#     "intent_sentence": str,
#     "target_labels": [str, ...],
#     "service_labels": [str, ...],
#     "confidence": float(0-1),
#     "followup": str or None
#   }
#   ※ UI には出さず、LLMが生成したメッセージはログにINFOで出力する
# --------------------------------------------------------------------
class IntentBuilder:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def build(self, user_query: str) -> Dict[str, Any]:
        system_prompt = (
            "あなたは自治体サービス検索の案内係です。"
            "ユーザー質問から、検索で使う代表的な1文（短く具体的）を作成し、"
            "対象者ラベル・サービスラベルを付与し、確信度(0-1)を数値で返し、"
            "不明瞭なら追質問を1つだけ生成してください。"
            "必ずJSONのみを出力し、キーは intent_sentence, target_labels, service_labels, confidence, followup にしてください。"
            "意図が明確なら followup は null にしてください。"
            "\n\n"
            "【対象者ラベル候補】\n"
            "  乳幼児（0～2歳）, 未就学児（3歳〜小学校入学前）, 小学生, 中学生, 高校生, 大学生, "
            "  保護者, 社会人, 高齢者, 障がい者, 事業者, 男性, 女性, どなたでも利用・参加可能, その他（該当が不明な場合）\n\n"
            "【サービスラベル候補】\n"
            "  補助金・助成金, 住まい・住宅支援, ペット・動物愛護, 水道・上下水道, 公園・緑地・レクリエーション, "
            "  意見・要望・苦情受付, 健康・医療, 福祉・介護, 子育て・教育, 雇用・就労支援, 市民生活・手続き, 防災・災害対応, "
            "  環境・ごみ・リサイクル, まちづくり・都市整備, 産業・事業者支援, 文化・スポーツ, 交通・移動支援, 移住・定住促進, "
            "  男女共同参画・人権・相談, 行政運営・計画・評価, 選挙・政治参加, デジタル・IT関連, 消費生活・トラブル対応, その他\n"
        )
        user_prompt = (
            f"ユーザー質問: {user_query}\n"
            "注意: intent_sentence は実際に検索にかける代表文です。可能なら手続名や補助金名を具体化してください。"
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},  # JSON を強制
        )
        content = (resp.choices[0].message.content or "").strip()

        # 解析とログ出力
        try:
            data = json.loads(content)
        except Exception as e:
            logger.warning("IntentBuilder JSON parse failed: %s | raw=%s", e, content[:300].replace("\n", " "))
            # フォールバック（followup をログに出すため定型文を入れる）
            data = {
                "intent_sentence": user_query,
                "target_labels": [],
                "service_labels": [],
                "confidence": 0.0,
                "followup": "どのような種類のサービス（例：補助金、手続き、相談、求人等）をお探しですか？",
            }

        # 既定値補強
        intent_sentence = data.get("intent_sentence") or user_query
        target_labels = data.get("target_labels", []) or []
        service_labels = data.get("service_labels", []) or []
        confidence = data.get("confidence", 0.0)
        followup = data.get("followup", None)

        # 型整形
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

        # ★ ここで「UIには出さずに」ログ出力する
        # followup があればそのまま、なければ「検索用にこう解釈しました：intent_sentence」
        if followup:
            logger.info("[IntentBuilder msg] %s", str(followup))
        else:
            logger.info('[IntentBuilder msg] 検索用にこう解釈しました：「%s」', intent_sentence)

        # 参考ログ（デバッグ用に要約も出す）
        logger.info(
            "IntentBuilder summary: sentence='%s' conf=%.2f targets=%s services=%s",
            intent_sentence, confidence, target_labels, service_labels
        )

        # 呼び出し側が使えるように返却（UI 表示はしない）
        return {
            "intent_sentence": intent_sentence,
            "target_labels": target_labels,
            "service_labels": service_labels,
            "confidence": confidence,
            "followup": followup,
        }

