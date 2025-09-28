# app/conversation_graph.py
from typing import List, Optional, TypedDict
import logging
from langgraph.graph import StateGraph, END

from llm_utils import label_question, IntentBuilder

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for LangGraph conversation flow."""
    question: str
    target_labels: List[str]
    service_labels: List[str]
    action: str
    followup: Optional[str]
    # フィードバック関連
    feedback: Optional[str]        # "good"/"yes"/"positive" or "bad"/"no"/"negative"
    refine_hint: Optional[str]
    # 意図確定フロー
    intent_sentence: Optional[str]
    intent_confidence: Optional[float]


# -------------------------
# （任意だが有効）簡易ヒューリスティック補正
# -------------------------
def _heuristic_labels(q: str) -> tuple[list, list]:
    """単語ヒットで最低限のラベルを補強（LLMが外した際の保険）"""
    tg: list = []
    sv: list = []
    if "定年" in q or "退職" in q:
        tg.append("高齢者")
    if "求人" in q or "就職" in q or "仕事" in q:
        sv.append("雇用・就労支援")
    return tg, sv


# -------------------------
# 既存: ラベル付け → 対象者が不明なら ask
# -------------------------
def classify_node(state: GraphState) -> GraphState:
    logger.info("ClassifyNode: question=%s", state["question"])
    labels = label_question(state["question"])
    state["target_labels"] = labels.get("target_labels", []) or []
    state["service_labels"] = labels.get("service_labels", []) or []

    # 簡易ヒューリスティック補正をマージ
    ht, hs = _heuristic_labels(state["question"])
    if ht:
        state["target_labels"] = list({*state["target_labels"], *ht})
    if hs:
        state["service_labels"] = list({*state["service_labels"], *hs})

    logger.info(
        "ClassifyNode: targets=%s services=%s",
        state["target_labels"], state["service_labels"]
    )
    return state


def decide_next(state: GraphState) -> GraphState:
    targets = state.get("target_labels", [])
    # 対象者が欠損 or 「その他」を含む → まず対象者を確定
    need_target = (not targets) or any("その他" in t for t in targets)
    if need_target:
        state["action"] = "ask"
        state["followup"] = (
            "サービスを利用する対象者を教えてください。\n"
            "提供サービス、補助金や支援の対象を正確にご案内するために、サービス提供対象者のご家族や世帯の状況を教えていただけますか？\n"
            "例: 3歳の子がいる母親、高校生の子を持つ父親、65歳以上の一人暮らし など"
        )
        logger.info("DecideNode: action=%s followup=%s", state["action"], bool(state.get("followup")))
    else:
        # 対象者が明確 → 意図確定フェーズへ
        state["action"] = "intent"
        state["followup"] = None
        logger.info("DecideNode: action=%s followup=%s", state["action"], bool(state.get("followup")))
    return state


# -------------------------
# 追加: 意図確定ノード
# -------------------------
_intent_builder = IntentBuilder()

def intent_node(state: GraphState) -> GraphState:
    res = _intent_builder.build(state["question"])
    state["intent_sentence"] = res.get("intent_sentence") or state["question"]
    # LLM側が返したラベルを優先（空なら既存値を残す）
    state["target_labels"] = res.get("target_labels") or state.get("target_labels", [])
    state["service_labels"] = res.get("service_labels") or state.get("service_labels", [])
    try:
        state["intent_confidence"] = float(res.get("confidence") or 0.0)
    except Exception:
        state["intent_confidence"] = 0.0
    state["followup"] = res.get("followup")
    logger.info(
        "IntentNode: sentence=%s, conf=%.2f, targets=%s, services=%s",
        state["intent_sentence"], state["intent_confidence"]
        if state["intent_confidence"] is not None else -1.0,
        state["target_labels"], state["service_labels"]
    )
    return state


def intent_decide_node(state: GraphState) -> GraphState:
    conf = float(state.get("intent_confidence") or 0.0)
    logger.info(
        "IntentDecide: conf=%.2f labels(t=%s,s=%s)",
        conf, state.get("target_labels"), state.get("service_labels")
    )
    if conf >= 0.7:
        state["action"] = "search_intent"
        state["followup"] = None
        logger.info("IntentDecide: action=%s conf=%.2f", state["action"], conf)
    else:
        state["action"] = "ask_intent"
        if not state.get("followup"):
            state["followup"] = "どのような種類のサービス（例：補助金、手続き、相談、求人等）をお探しですか？"
        logger.info("IntentDecide: action=%s conf=%.2f", state["action"], conf)
    return state


# -------------------------
# グラフ定義（条件分岐ルーティングを厳密化）
# -------------------------
_graph = StateGraph(GraphState)
_graph.add_node("classify", classify_node)
_graph.add_node("decide", decide_next)
_graph.add_node("intent", intent_node)
_graph.add_node("intent_decide", intent_decide_node)
_graph.set_entry_point("classify")
_graph.add_edge("classify", "decide")

# decide の結果に応じて次ノードを条件分岐
def _route_from_decide(state: GraphState):
    nxt = "intent" if state.get("action") == "intent" else "__end__"
    logger.info("Router(decide): action=%s -> next=%s", state.get("action"), nxt)
    return nxt

_graph.add_conditional_edges(
    "decide",
    _route_from_decide,
    {
        "intent": "intent",
        "__end__": END,   # action=ask のときはここで終了（main.py が追質問を提示）
    },
)

_graph.add_edge("intent", "intent_decide")
_graph.add_edge("intent_decide", END)

workflow = _graph.compile()


# -------------------------
# Feedback-only workflow
# -------------------------
def _build_refine_followup(state: GraphState) -> str:
    tgt = state.get("target_labels") or []
    svc = state.get("service_labels") or []
    base = "（対象者／サービス種別／オンライン手続き可否／費用・助成／時期・締切／地域・施設）"
    if not tgt:
        return (
            "結果が期待と異なるようです。まず、サービスの対象者をもう少し詳しく教えてください。\n"
            "例：3歳児の保護者／高校生の保護者／65歳以上の一人暮らし など\n"
            f"併せて、優先したい条件があれば教えてください。{base}"
        )
    if not svc:
        return (
            "結果が期待と異なるようです。探しているサービスの種類（補助金・助成金、手続き、イベント等）を教えてください。\n"
            f"また、優先したい条件があれば教えてください。{base}"
        )
    return f"どの点が期待と異なっていましたか？\n優先したい条件があれば教えてください。{base}"


def feedback_node(state: GraphState) -> GraphState:
    fb = (state.get("feedback") or "").lower()
    if fb in {"good", "yes", "positive", "satisfied"}:
        state["action"] = "end"
        state["followup"] = None
        return state
    state["action"] = "ask"
    state["followup"] = _build_refine_followup(state)
    return state


_feedback_graph = StateGraph(GraphState)
_feedback_graph.add_node("feedback", feedback_node)
_feedback_graph.set_entry_point("feedback")
_feedback_graph.add_edge("feedback", END)

feedback_workflow = _feedback_graph.compile()

