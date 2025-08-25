from __future__ import annotations
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from llm_utils import label_question


class GraphState(TypedDict):
    """State for LangGraph conversation flow."""
    question: str
    target_labels: List[str]
    service_labels: List[str]
    action: str
    followup: Optional[str]


def classify_node(state: GraphState) -> GraphState:
    """Classify user question into target and service labels."""
    labels = label_question(state["question"])
    state["target_labels"] = labels.get("target_labels", [])
    state["service_labels"] = labels.get("service_labels", [])
    return state


def decide_next(state: GraphState) -> GraphState:
    """Decide whether to ask for target info or proceed to search."""
    targets = state.get("target_labels", [])
    # ask when target is missing or labeled as other/unknown
    if (not targets) or any("その他" in t for t in targets):
        state["action"] = "ask"
        state["followup"] = (
            "サービスを利用する対象者を教えてください。\n"
            "例: 乳幼児, 未就学児, 小学生, 中学生, 高校生, 大学生, 保護者, 社会人, 高齢者, 障がい者, 事業者, 男性, 女性, どなたでも利用・参加可能"
        )
    else:
        state["action"] = "search"
        state["followup"] = None
    return state


# build LangGraph workflow
_graph = StateGraph(GraphState)
_graph.add_node("classify", classify_node)
_graph.add_node("decide", decide_next)
_graph.set_entry_point("classify")
_graph.add_edge("classify", "decide")
_graph.add_edge("decide", END)

workflow = _graph.compile()
