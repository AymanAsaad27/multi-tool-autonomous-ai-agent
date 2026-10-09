from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    task: str

    plan: list[str]

    step_idx: int

    messages: Annotated[list, add_messages]

    results: list[str]

    # Current-step retry counter.
    # This resets after a step succeeds.
    retries: int

    feedback: str

    # Current-step tool-call counter.
    # This resets when a new step begins.
    tool_calls_made: int

    # Cumulative counters used for benchmark evaluation.
    total_tool_calls: int

    total_retries: int

    failed_steps: int

    final_answer: str


def initial_state(task: str) -> AgentState:

    return {
        "task": task,
        "plan": [],
        "step_idx": 0,
        "messages": [],
        "results": [],
        "retries": 0,
        "feedback": "",
        "tool_calls_made": 0,
        "total_tool_calls": 0,
        "total_retries": 0,
        "failed_steps": 0,
        "final_answer": "",
    }