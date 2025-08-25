import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_CURRENT_DIR)

from embed_utils import embed_text
from llm_utils import ServiceSelector
from catalog_utils import CatalogSearchEngine
from conversation_graph import workflow

load_dotenv()

st.set_page_config(page_title="自治体サービス案内チャット", page_icon="🏛️")

# keep chat history between reruns
if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, msg)]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

searcher = CatalogSearchEngine()
selector = ServiceSelector()

# --- Render chat history (moved before form to ensure it's shown even if rerun/stop) ---
for role, msg in st.session_state.history:
    avatar = "🧑‍💻" if role == "user" else "🤖"
    label  = "利用者" if role == "user" else "案内"
    st.chat_message(f"{avatar} {label}", avatar=avatar).markdown(msg)

# --- Chat input form ---
with st.form("chat_form", clear_on_submit=True):
    user_msg = st.text_input("なんでも質問してください")
    submitted = st.form_submit_button("送信")

# --- When user sends a message ---
if submitted and user_msg:
    # 1) store user message
    st.session_state.history.append(("user", user_msg))
    logger.info("Received user message: %s", user_msg)

    # combine with pending question if we previously asked for target info
    combined_question = f"{st.session_state.pending_question} {user_msg}".strip()
    logger.info(
        "Combined question: '%s' (pending='%s')",
        combined_question,
        st.session_state.pending_question,
    )

    # 2) classify using LangGraph workflow
    try:
        state = workflow.invoke({"question": combined_question, "target_labels": [], "service_labels": []})
        logger.info("Workflow output: %s", state)
    except Exception:
        logger.exception("label_question failed")
        st.session_state.history.append(("assistant", "内部エラーが発生しました。時間を置いて再度お試しください。"))
        st.rerun()

    if state["action"] == "ask":
        # ask user to specify target
        st.session_state.pending_question = combined_question
        logger.info("Action=ask pending_question set to: %s", st.session_state.pending_question)
        st.session_state.history.append(("assistant", state["followup"]))
        st.rerun()

    st.session_state.pending_question = ""
    target_labels = state.get("target_labels", [])
    service_labels = state.get("service_labels", [])
    logger.info(
        "検索を実行: 対象者ラベル=%s サービスラベル=%s",
        target_labels,
        service_labels,
    )

    # 3) filter catalog by labels
    filtered_df = searcher.filter_by_labels(target_labels, service_labels)

    # ログ出力: フィルター後の件数
    logger.info(f"フィルター後のサービス件数: {len(filtered_df)}件")

    # 4) BERT embed + similarity ranking (top 50)
    query_vec = embed_text(combined_question)
    ranked_df = searcher.rank(filtered_df, query_vec, top_n=50)

    # 5) craft assistant reply using LLM selection
    if ranked_df.empty:
        assistant_reply = "すみません、該当する自治体サービスが見つかりませんでした。別の表現でお試しください。"
    else:
        candidates = [
            {"title": row["タイトル"], "url": row["URL"]["items"]}
            for _, row in ranked_df.iterrows()
        ]
        try:
            recs = selector.recommend(combined_question, candidates)
        except Exception:
            logger.exception("recommend failed")
            recs = []
        if recs:
            services = "\n".join(
                f"- **{r['title']}** ({r['url']})" for r in recs
            )
            assistant_reply = "おすすめのサービスはこちらです:\n" + services
        else:
            top_rows = ranked_df.head(3)
            services = "\n".join(
                f"- **{row['タイトル']}** ({row['URL']['items']})" for _, row in top_rows.iterrows()
            )
            assistant_reply = "以下のサービスが見つかりました:\n" + services

    st.session_state.history.append(("assistant", assistant_reply))
    st.rerun()
