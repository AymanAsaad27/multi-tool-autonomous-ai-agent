from langchain_core.messages import AIMessage

from agent.state import initial_state
from agent.text import text_of


def test_initial_state():
    state = initial_state("Test task")

    assert state["task"] == "Test task"
    assert state["plan"] == []
    assert state["step_idx"] == 0
    assert state["results"] == []
    assert state["retries"] == 0
    assert state["feedback"] == ""
    assert state["tool_calls_made"] == 0
    assert state["failed_steps"] == 0
    assert state["final_answer"] == ""


def test_text_of_string_content():
    message = AIMessage(content="Hello agent")

    assert text_of(message) == "Hello agent"


def test_text_of_gemini_style_content():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "Hello from Gemini",
            }
        ]
    )

    assert text_of(message) == "Hello from Gemini"