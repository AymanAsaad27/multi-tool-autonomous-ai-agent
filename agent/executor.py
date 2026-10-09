from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException

from agent.config import get_llm
from agent.tools import TOOLS


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

8. For database tasks, if the previous attempt returned "No results found"
   or a database/SQL error, you MUST change your approach.

   Do NOT repeat the exact same SQL query.

   First use query_database to inspect the schema, for example:
   PRAGMA table_info(notes)

   Then use the discovered table and column names to construct a new,
   corrected SQL query.

   Use the previous verifier feedback to understand why the first query
   failed.

"""


# Create the base language model.
llm = get_llm()

# Give Gemini access to our six tools.
executor_llm = llm.bind_tools(TOOLS)


def execute_step(
    task: str,
    step: str,
    previous_results: list[str] | None = None,
    feedback: str = "",
):
    """
    Execute one planned step.

    Returns the messages produced during this execution.
    """

    previous_results = previous_results or []

    message = f"""
Overall task:
{task}

Current step:
{step}
"""

    if previous_results:
        message += """

Results from earlier completed steps:
"""

        for i, result in enumerate(previous_results, start=1):
            message += f"{i}. {result}\n"

    if feedback:
        message += f"""

The previous attempt at this step failed.

Verifier feedback:
{feedback}

Try a different approach and correct the problem.
"""

    message += """

Complete ONLY the current step.
Use the available tools when necessary.
When finished, give a short summary of the result.
"""

    messages = [
        SystemMessage(content=EXECUTOR_PROMPT),
        HumanMessage(content=message),
    ]

    try:
        response = executor_llm.invoke(messages)
        return response

    except Exception as e:
        print(
            f"\n[Executor warning] Gemini request failed: "
            f"{type(e).__name__}: {e}\n"
        )
        raise