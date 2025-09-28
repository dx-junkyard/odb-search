# app/main.py
import logging
from typing import Dict, Any, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from conversation_graph import workflow, feedback_workflow
from catalog_utils import CatalogSearchEngine
from embed_utils import embed_text
from llm_utils import ServiceSelector

# -----------------------------------------------------------------------------
# 初期設定
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="自治体サービス検索", layout="wide")
st.title("自治体サービス検索システム")

# -----------------------------------------------------------------------------
# セッション状態
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False
if "refine_loops" not in st.session_state:
    st.session_state.refine_loops = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_labels" not in st.session_state:
    st.session_state.last_labels = ([], [])

# -----------------------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------------------
def extract_url(val) -> str:
    """service_catalog.json の URL 形式ゆらぎに耐える抽出器"""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        if val and isinstance(val[0], str):
            return val[0].strip()
        return ""
    if isinstance(val, dict):
        for k in ("items", "item", "url", "URL", "link", "links"):
            if k in val:
                v = val[k]
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return v[0].strip()
    return ""

def series_get(row, key, default=None):
    try:
        if hasattr(row, "get"):
            return row.get(key, default)
        return row[key] if key in row else default
    except Exception:
        return default

def row_to_title_url(row) -> Tuple[str, str]:
    title = series_get(row, "タイトル", "")
    url_field = series_get(row, "URL", "")
    url = extract_url(url_field)
    title = str(title).strip() if title is not None else ""
    return title, url

# -----------------------------------------------------------------------------
# 検索エンジン/セレクタ
# -----------------------------------------------------------------------------
searcher = CatalogSearchEngine()
selector = ServiceSelector()

# -----------------------------------------------------------------------------
# チャット履歴の表示
# -----------------------------------------------------------------------------
for role, msg in st.session_state.history:
    st.chat_message("user" if role == "user" else "assistant").write(msg)

# -----------------------------------------------------------------------------
# 入力フォーム
# -----------------------------------------------------------------------------
with st.form("chat-form", clear_on_submit=True):
    user_msg = st.text_input("質問を入力してください", "")
    submitted = st.form_submit_button("送信")

# -----------------------------------------------------------------------------
# 送信処理
# -----------------------------------------------------------------------------
if submitted and user_msg:
    # 1) store user message
    st.session_state.history.append(("user", user_msg))
    st.chat_message("user").write(user_msg)  # ★ その場で描画
    logger.info("Received user message: %s", user_msg)

    # combine with pending question if we previously asked for target/intent info
    combined_question = f"{st.session_state.pending_question} {user_msg}".strip()
    logger.info("Combined question: '%s' (pending='%s')",
                combined_question, st.session_state.pending_question)
    # 直近入力を保存
    st.session_state.last_query = combined_question

    # 2) LangGraph workflow
    try:
        state = workflow.invoke(
            {"question": combined_question, "target_labels": [], "service_labels": []}
        )
        logger.info("Workflow output: %s", state)
    except Exception:
        logger.exception("workflow failed")
        st.session_state.history.append(
            ("assistant", "内部エラーが発生しました。時間を置いて再度お試しください。")
        )
        st.rerun()

    # 分岐1: 対象者が不明 -> 先に対象者確定
    if state["action"] == "ask":
        st.session_state.pending_question = combined_question
        logger.info("Action=ask pending_question='%s'", st.session_state.pending_question)
        st.session_state.history.append(("assistant", state["followup"]))
        st.rerun()

    # 分岐2: 意図が不明 -> 意図確定のための1問
    if state["action"] == "ask_intent":
        st.session_state.pending_question = combined_question
        logger.info("Action=ask_intent pending_question='%s'", st.session_state.pending_question)
        st.session_state.history.append(("assistant", state.get("followup") or "もう少し詳しく教えてください。"))
        st.rerun()

    # ここに来るのは search_intent（意図確度高い） or intent確定後の通常検索
    st.session_state.pending_question = ""
    intent_sentence = state.get("intent_sentence") or combined_question
    target_labels = state.get("target_labels", [])
    service_labels = state.get("service_labels", [])
    logger.info("SEARCH intent_sentence='%s'", intent_sentence)
    logger.info("検索を実行: 対象者ラベル=%s サービスラベル=%s", target_labels, service_labels)

    st.session_state.last_labels = (target_labels, service_labels)
    st.session_state.last_query = intent_sentence  # 検索に使った文として保存

    # 3) カタログをラベルでフィルタ
    filtered_df = searcher.filter_by_labels(target_labels, service_labels)
    logger.info("フィルター後のサービス件数: %s件", len(filtered_df))

    # 4) 埋め込み + 類似度上位
    query_vec = embed_text(intent_sentence)
    ranked_df = searcher.rank(filtered_df, query_vec, top_n=50)

    # 5) 推薦/表示
    header = f"**検索用にこう解釈しました：**「{intent_sentence}」"
    try:
        if ranked_df is None or len(ranked_df) == 0:
            assistant_reply = header + "\n\nすみません、該当する自治体サービスが見つかりませんでした。別の表現でお試しください。"
        else:
            # 候補を安全に構築
            candidates = []
            for _, row in ranked_df.iterrows():
                t, u = row_to_title_url(row)
                if t:
                    candidates.append({"title": t, "url": u})

            # LLM 推薦（失敗してもフォールバックで上位3件を出す）
            recs = []
            try:
                recs = selector.recommend(intent_sentence, candidates) or []
            except Exception:
                logger.exception("recommend failed; fall back to top-3")

            if recs:
                lines = []
                for r in recs:
                    t = str(r.get("title", "")).strip()
                    u = extract_url(r.get("url"))
                    if t:
                        lines.append(f"- **{t}** ({u})" if u else f"- **{t}**")
                assistant_reply = header + ("\n\nおすすめのサービスはこちらです:\n" + "\n".join(lines)
                                            if lines else "\n\nおすすめ候補を生成できませんでした。")
            else:
                # フォールバック: 上位3件
                top_rows = ranked_df.head(3)
                lines = []
                for _, row in top_rows.iterrows():
                    t, u = row_to_title_url(row)
                    if t:
                        lines.append(f"- **{t}** ({u})" if u else f"- **{t}**")
                assistant_reply = header + ("\n\n以下のサービスが見つかりました:\n" + "\n".join(lines)
                                            if lines else "\n\n候補が生成できませんでした。")

        st.session_state.history.append(("assistant", assistant_reply))
        st.chat_message("assistant").write(assistant_reply)  # ★ その場で描画
        logger.info("Rendered %d services", assistant_reply.count("\n"))
    except Exception:
        logger.exception("Rendering services failed")
        st.session_state.history.append(
            ("assistant", header + "\n\n候補の表示でエラーが発生しました。入力条件を少し変えて再度お試しください。")
        )

    # 検索結果に対するフィードバックを受け付ける
    st.session_state.awaiting_feedback = True
    # rerunせず、このフレームでUIを描画

# -----------------------------------------------------------------------------
# 検索結果へのフィードバック処理
# -----------------------------------------------------------------------------
if st.session_state.get("awaiting_feedback", False):
    st.divider()
    st.info("この結果は役立ちましたか？")
    col1, col2 = st.columns(2)
    with col1:
        fb_yes = st.button("はい、終了する", key="fb_yes")
    with col2:
        fb_no = st.button("いいえ、条件を絞り込む", key="fb_no")

    if fb_yes:
        # 終了
        st.session_state.history.append(("assistant", "ご利用ありがとうございました。別の質問もどうぞ。"))
        st.session_state.awaiting_feedback = False
        st.session_state.refine_loops = 0
        st.rerun()

    if fb_no:
        # ループ上限チェック
        st.session_state.refine_loops += 1
        if st.session_state.refine_loops >= 3:
            st.session_state.history.append(
                ("assistant", "うまく見つからないようです。担当窓口や有人チャットをご案内できます。続けますか？「続ける」または「終了」と入力してください。")
            )
            st.session_state.awaiting_feedback = False
            st.rerun()

        # フィードバックに基づく再絞り込み質問
        try:
            fb_state = feedback_workflow.invoke(
                {
                    "question": st.session_state.last_query,
                    "target_labels": st.session_state.last_labels[0],
                    "service_labels": st.session_state.last_labels[1],
                    "feedback": "bad",
                    "refine_hint": None,
                }
            )
            followup = fb_state.get("followup") or (
                "より具体的に条件を教えてください（対象者／サービス種別／オンライン手続き可否／費用・助成／時期・締切／地域・施設）。"
            )
        except Exception:
            logger.exception("feedback_workflow failed")
            followup = "より具体的に条件を教えてください（対象者／サービス種別／オンライン手続き可否／費用・助成／時期・締切／地域・施設）。"

        st.session_state.pending_question = st.session_state.last_query  # 直前の意図文を保持
        logger.info("ASK(by_feedback): pending='%s'", st.session_state.pending_question)
        st.session_state.history.append(("assistant", followup))
        st.session_state.awaiting_feedback = False
        st.rerun()

