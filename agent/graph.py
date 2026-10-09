from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent.config import get_llm
from agent.state import AgentState
from agent.text import text_of
from agent.tools import TOOLS


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_RETRIES = 2
MAX_TOOL_CALLS = 8


# ---------------------------------------------------------
# Structured output model for the verifier
# ---------------------------------------------------------

class StepCheck(BaseModel):
    """Structured result returned by the verifier."""

    success: bool = Field(
        description="Whether the current step was successfully completed."
    )

    summary: str = Field(
        description="Short summary explaining why the step succeeded or failed."
    )


# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------

EXECUTOR_PROMPT = """
You are the execution module of an autonomous AI agent.

Your job is to complete ONE step at a time using the available tools.

Rules:

1. Use tools to obtain real information.
2. Never guess facts, numbers, dates, database contents, or search results.
3. Choose the appropriate tool for the current step.
4. If a tool returns an error, carefully read the error and try a different
   approach when possible.
5. Do not attempt to complete other steps from the overall task.
6. When the current step is complete, provide a short summary of the
   concrete result.
7. Include important numbers, names, dates, or values in the summary.
"""

VERIFIER_PROMPT = """
You are a strict verifier for an autonomous AI agent.

Your job is to determine whether ONE specific step was completed successfully.

You will receive:

1. CURRENT STEP
2. EVIDENCE FOR THIS ATTEMPT ONLY

Evaluate ONLY the current step.

Do NOT use information from previous steps or previous attempts.

Mark success=false if ANY of the following is true:

- The current step was not completed.
- A required tool was not used.
- The tool returned an error.
- The result is missing.
- The result is unsupported by the evidence.
- The agent only described what it plans to do.
- The agent performed a different step instead of completing the current step.
- The agent claims success without evidence.

Mark success=true when:

1. The current step has actually been completed.
2. The required tool was used when a tool was necessary.
3. The tool result provides evidence that the step was completed.
4. The executor's statement is consistent with the tool result.

Important:

- Tool results are the strongest evidence.
- The executor does NOT need to repeat the tool result word-for-word.
- If the executor accurately communicates the tool result in natural
  language, that is sufficient.
- Do not require unnecessary formatting.
- Do not confuse a previous step with the current step.
- Do not confuse a previous retry with the current attempt.
- If the required tool succeeded and its result directly answers the
  current step, mark the step successful.
- Give a short, concrete summary of the evidence.
- Be consistent: apply the same evidence-based criteria to every attempt.
- If the required tool was called and its result directly answers the
  current step, do not reject the step because of wording, formatting,
  or a missing explanation in the executor's response.
- For calculations, compare the requested calculation with the actual
  tool result. Do not reject a correct numeric result without a specific
  evidence-based reason.
- Never mark a step successful if the tool result contradicts the
  requested result or indicates an error.
Return:
- success=true when the current step is genuinely complete.
- success=false when the current step genuinely needs another attempt.
"""


SYNTH_PROMPT = """
You write the final answer for the user.

Your answer MUST be grounded ONLY in the evidence provided below.

IMPORTANT:
- The TOOL EVIDENCE is authoritative.
- Do not invent names, dates, numbers, descriptions, IDs, or other facts.
- Do not replace missing information with examples or guesses.
- Preserve important values from the tool evidence accurately.
- If the user asks for a list, use the actual items from the tool evidence.
- Do not create a new list from general knowledge.
- Do not claim that information exists unless it appears in the evidence.
- Completed step summaries may describe the evidence, but they are NOT
  a substitute for the actual tool output.
- If tool evidence is available, prefer it over the step summary.
- If the evidence is insufficient to answer the task, say that the
  available evidence is insufficient.

Be concise while still answering the user's request completely.
"""


# ---------------------------------------------------------
# LLM setup
# ---------------------------------------------------------

llm = get_llm()

verifier_llm = llm.with_structured_output(StepCheck)



# ---------------------------------------------------------
# Planner
# ---------------------------------------------------------

def planner(state: AgentState):
    """
    Convert the user's task into a short ordered plan.

    The actual planning logic is implemented in agent.planner.py.
    """

    from agent.planner import create_plan

    plan = create_plan(state["task"])

    return {
        "plan": plan,
        "messages": [
            SystemMessage(content=EXECUTOR_PROMPT)
        ],
    }


# ---------------------------------------------------------
# Begin one step
# ---------------------------------------------------------

def begin_step(state: AgentState):
    """
    Prepare the executor for the current step.
    """

    i = state["step_idx"]

    message = (
        f"Overall task:\n"
        f"{state['task']}\n\n"
        f"Current step {i + 1}/{len(state['plan'])}:\n"
        f"{state['plan'][i]}\n"
    )

    if state["results"]:
        message += (
            "\nResults from earlier completed steps:\n"
            + "\n".join(
                f"{n + 1}. {result}"
                for n, result in enumerate(state["results"])
            )
            + "\n"
        )

    if state["feedback"]:
        message += (
        "\nThe previous attempt at this step failed.\n"
        f"Verifier feedback:\n{state['feedback']}\n\n"
        "You MUST use a different approach.\n"
    )

    if "query_database" in state["plan"][i].lower():
        message += (
        "\nDATABASE INSTRUCTIONS:\n"
        "Use query_database for this step.\n"
        "\n"
        "If this is the first attempt:\n"
        "1. Inspect the relevant table schema if necessary.\n"
        "2. Construct a SQL query that directly answers the user's "
        "actual requirement.\n"
        "\n"
        "If the previous attempt failed:\n"
        "1. Do NOT repeat the exact same SQL query.\n"
        "2. Read the verifier feedback carefully.\n"
        "3. Inspect the actual table data when necessary.\n"
        "4. Construct a genuinely different query that addresses "
        "the failure.\n"
        "5. IMPORTANT: If you describe a SQL query in your response, "
        "you MUST actually execute it using query_database.\n"
        "\n"
        "For content-based questions, do not assume the title and "
        "content contain the same information. Inspect the actual "
        "content values when necessary.\n"
        "\n"
        "Do not merely propose SQL. Execute the SQL using the tool."
    )

    message += (
        "\nComplete ONLY this step.\n"
        "Use the available tools when necessary.\n"
        "When finished, provide a short summary of the result."
    )

    return {
    "messages": [
        HumanMessage(content=message)
    ],
    "tool_calls_made": 0,
}


# ---------------------------------------------------------
# Executor
# ---------------------------------------------------------

def executor(state: AgentState):
    if state["tool_calls_made"] >= MAX_TOOL_CALLS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Stopped: tool-call budget for this step "
                        "was exceeded."
                    )
                )
            ]
        }

    current_step = state["plan"][state["step_idx"]]
    step_lower = current_step.lower()

    if "get_current_datetime" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "get_current_datetime"]

    elif "web_search" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "web_search"]

    elif "run_python" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "run_python"]

    elif "query_database" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "query_database"]

    elif "add_calendar_event" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "add_calendar_event"]

    elif "list_calendar_events" in step_lower:
        allowed_tools = [tool for tool in TOOLS if tool.name == "list_calendar_events"]

    else:
        allowed_tools = TOOLS

    current_executor_llm = llm.bind_tools(allowed_tools)

    try:
        ai = current_executor_llm.invoke(state["messages"])
    except Exception as e:
        print(
            "\n[Executor warning] LLM request failed: "
            f"{type(e).__name__}: {e}\n"
        )
        raise

    tool_calls = getattr(ai, "tool_calls", [])

    return {
    "messages": [ai],

    # Current-step tool-call count.
    "tool_calls_made": (
        state["tool_calls_made"] + len(tool_calls)
    ),

    # Cumulative benchmark metric.
    "total_tool_calls": (
        state["total_tool_calls"] + len(tool_calls)
    ),
}

# ---------------------------------------------------------
# Decide whether executor needs a tool
# ---------------------------------------------------------

def route_executor(state: AgentState):
    """
    If the executor requested one or more tools, execute them.

    Otherwise, send the result to the verifier.
    """

    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "tools"

    return "verify"

# ---------------------------------------------------------
# Verifier
# ---------------------------------------------------------

def verify(state: AgentState):
    """
    Verify whether the current step was completed successfully.

    The verifier only examines messages belonging to the
    current attempt. This prevents evidence from previous
    steps or previous retries from contaminating verification.
    """

    i = state["step_idx"]
    step = state["plan"][i]

    messages = state["messages"]

    # -----------------------------------------------------
    # Find the most recent "Current step" HumanMessage.
    #
    # begin_step() creates one of these every time a step
    # starts or retries.
    # -----------------------------------------------------

    current_attempt_start = 0

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]

        if isinstance(message, HumanMessage):
            content = text_of(message)

            if "Current step" in content:
                current_attempt_start = index
                break

    # Only inspect messages belonging to this attempt.
    recent_messages = messages[current_attempt_start:]

    evidence_parts = []

    for message in recent_messages:
        message_type = type(message).__name__
        content = text_of(message)

        if content:
            evidence_parts.append(
                f"[{message_type}]\n{content}"
            )

        tool_calls = getattr(message, "tool_calls", [])

        for tool_call in tool_calls:
            evidence_parts.append(
                f"[Tool call]\n"
                f"Name: {tool_call['name']}\n"
                f"Arguments: {tool_call['args']}"
            )

    evidence = "\n\n".join(evidence_parts)

    # -----------------------------------------------------
    # Ask the verifier to evaluate ONLY this step/attempt.
    # -----------------------------------------------------

    check = verifier_llm.invoke(
        [
            SystemMessage(content=VERIFIER_PROMPT),
            HumanMessage(
                content=(
                    f"CURRENT STEP:\n{step}\n\n"
                    f"EVIDENCE FOR THIS ATTEMPT ONLY:\n"
                    f"{evidence}\n\n"
                    "Evaluate ONLY the current step. "
                    "Ignore all previous steps and previous attempts."
                )
            ),
        ]
    )

    # -----------------------------------------------------
    # Successful step
    # -----------------------------------------------------

    if check.success:
        return {
            "results": state["results"] + [check.summary],
            "step_idx": i + 1,
            "retries": 0,
            "feedback": "",
        }

    # -----------------------------------------------------
    # Retry current step
    # -----------------------------------------------------

    if state["retries"] < MAX_RETRIES:
        return {
            "retries": state["retries"] + 1,
            "total_retries": state["total_retries"] + 1,
            "feedback": check.summary,
        }

    # -----------------------------------------------------
    # Final failure after all retries
    # -----------------------------------------------------

    return {
        "results": state["results"]
        + [f"(FAILED) {check.summary}"],
        "step_idx": i + 1,
        "retries": 0,
        "feedback": "",
        "failed_steps": state["failed_steps"] + 1,
    }
# ---------------------------------------------------------
# Decide what happens after verification
# ---------------------------------------------------------

def route_verify(state: AgentState):
    """
    Continue to the next step or finish the task.
    """

    if state["step_idx"] >= len(state["plan"]):
        return "synthesize"

    return "begin_step"


# ---------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------

def synthesize(state: AgentState):
    """
    Combine verified step results and actual tool evidence
    into the final answer.
    """

    body = "\n".join(
        f"Step {n + 1} ({step}): {result}"
        for n, (step, result)
        in enumerate(
            zip(state["plan"], state["results"])
        )
    )

    evidence_parts = []

    for message in state["messages"]:
        content = text_of(message)

        if content:
            message_type = type(message).__name__

            evidence_parts.append(
                f"[{message_type}]\n{content}"
            )

    evidence = "\n\n".join(evidence_parts)

    output = llm.invoke(
        [
            SystemMessage(content=SYNTH_PROMPT),
            HumanMessage(
                content=(
                    f"Task:\n"
                    f"{state['task']}\n\n"

                    f"COMPLETED STEP RESULTS:\n"
                    f"{body}\n\n"

                    f"ACTUAL EXECUTION EVIDENCE:\n"
                    f"{evidence}\n\n"

                    "Write the final answer using only the "
                    "actual execution evidence and completed "
                    "step results above."
                )
            ),
        ]
    )

    return {
        "final_answer": text_of(output)
    }

# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    """
    Build and compile the autonomous agent graph.
    """

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner)
    graph.add_node("begin_step", begin_step)
    graph.add_node("executor", executor)

    graph.add_node(
        "tools",
        ToolNode(
            TOOLS,
            handle_tool_errors=True,
        ),
    )

    graph.add_node("verify", verify)
    graph.add_node("synthesize", synthesize)

    # ---------------------------------------------
    # Main flow
    # ---------------------------------------------

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_edge(
        "planner",
        "begin_step",
    )

    graph.add_edge(
        "begin_step",
        "executor",
    )

    # ---------------------------------------------
    # Executor → Tools or Verifier
    # ---------------------------------------------

    graph.add_conditional_edges(
        "executor",
        route_executor,
        {
            "tools": "tools",
            "verify": "verify",
        },
    )

    # Tool result goes back to executor
    graph.add_edge(
        "tools",
        "executor",
    )

    # ---------------------------------------------
    # Verifier → Retry or Next Step
    # ---------------------------------------------

    graph.add_conditional_edges(
        "verify",
        route_verify,
        {
            "begin_step": "begin_step",
            "synthesize": "synthesize",
        },
    )

    # ---------------------------------------------
    # Synthesizer → END
    # ---------------------------------------------

    graph.add_edge(
        "synthesize",
        END,
    )

    return graph.compile()