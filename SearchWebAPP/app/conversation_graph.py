from __future__ import annotations
from typing import List, Optional, TypedDict

import logging
from langgraph.graph import StateGraph, END

from llm_utils import label_question


logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for LangGraph conversation flow."""
    question: str
    target_labels: List[str]
    service_labels: List[str]
    action: str
    followup: Optional[str]


def classify_node(state: GraphState) -> GraphState:
    """Classify user question into target and service labels."""
    logger.info("ClassifyNode: question=%s", state["question"])
    labels = label_question(state["question"])
    state["target_labels"] = labels.get("target_labels", [])
    state["service_labels"] = labels.get("service_labels", [])
    logger.info(
        "ClassifyNode: target_labels=%s service_labels=%s",
        state["target_labels"],
        state["service_labels"],
    )
    return state


def decide_next(state: GraphState) -> GraphState:
    """Decide whether to ask for target info or proceed to search."""
    targets = state.get("target_labels", [])
    logger.info("DecideNode: targets=%s", targets)
    # ask when target is missing or labeled as other/unknown
    if (not targets) or any("その他" in t for t in targets):
        state["action"] = "ask"
        state["followup"] = (
            "サービスを利用する対象者を教えてください。\n"
            "提供サービス、補助金や支援の対象を正確にご案内するために、サービス提供対象者のご家族や世帯の状況を教えていただけますか？\n"
            "例: 3歳の子がいる母親、高校生の子を持つ父親、65歳以上の一人暮らし など"
        )
        logger.info("DecideNode: action=ask followup=%s", state["followup"])
    else:
        state["action"] = "search"
        state["followup"] = None
        logger.info("DecideNode: action=search")
    return state


# build LangGraph workflow
_graph = StateGraph(GraphState)
_graph.add_node("classify", classify_node)
_graph.add_node("decide", decide_next)
_graph.set_entry_point("classify")
_graph.add_edge("classify", "decide")
_graph.add_edge("decide", END)

workflow = _graph.compile()
