from langchain_core.messages import AIMessage

from agent.state import initial_state
from agent.tools import TOOLS
from agent.graph import build_graph


def test_graph_has_expected_nodes():
    app = build_graph()

    nodes = set(app.get_graph().nodes.keys())

    expected = {
        "__start__",
        "planner",
        "begin_step",
        "executor",
        "tools",
        "verify",
        "synthesize",
        "__end__",
    }

    assert expected.issubset(nodes)


def test_tools_have_expected_names():
    names = {tool.name for tool in TOOLS}

    assert names == {
        "get_current_datetime",
        "web_search",
        "run_python",
        "query_database",
        "add_calendar_event",
        "list_calendar_events",
    }


def test_initial_state_can_be_used_by_graph():
    state = initial_state(
        "Calculate 25 multiplied by 4 using Python."
    )

    assert state["task"] == (
        "Calculate 25 multiplied by 4 using Python."
    )

    assert state["step_idx"] == 0
    assert state["messages"] == []


def test_executor_tool_call_message_structure():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_python",
                "args": {
                    "code": "print(25 * 4)"
                },
                "id": "test-call-1",
            }
        ],
    )

    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]

    assert tool_call["name"] == "run_python"
    assert tool_call["args"]["code"] == "print(25 * 4)"
    assert tool_call["id"] == "test-call-1"

    from langgraph.prebuilt import ToolNode
