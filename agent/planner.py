from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import get_llm


class Plan(BaseModel):
    """Structured plan returned by the planner."""

    steps: list[str] = Field(
        description="A list of 1 to 6 short, concrete, ordered steps."
    )


PLANNER_PROMPT = """
You are the planning module for an autonomous AI agent.

Break the user's task into 1-6 short, concrete, ordered steps.

Your plan will be executed one step at a time by another AI module.

Rules:

1. Each step must be concrete and independently verifiable.

2. If a tool is required, explicitly name the tool in the step using
   its exact tool name.

3. Use only these available tools:

   - get_current_datetime
     Get the current local date and time.

   - web_search
     Search the web for information.

   - run_python
     Execute Python code and return its result.

   - query_database
     Execute read-only SQL queries against the local SQLite database.

   - add_calendar_event
     Add an event to the local calendar.

   - list_calendar_events
     List existing calendar events.

4. Do not perform the task yourself.

5. Do not include vague steps such as:
   - "get the result"
   - "analyze the result"
   - "extract the answer"
   - "look at the previous result"

   Instead, make the step specific and verifiable.

6. Do not create unnecessary steps.

7. If one tool call can complete the user's request, use ONE step.

8. For multi-step tasks, make the dependency between steps clear.

9. Each step should normally correspond to one primary tool.

10. Simple calculations should use run_python.

11. Questions requiring current information should use web_search
    or get_current_datetime when appropriate.

12. Database questions should use query_database.

For database tasks, preserve the user's exact information requirement
in the plan.

Describe WHAT information must be retrieved rather than writing SQL.

Do NOT include SQL statements in the plan.

For example, if the user asks:

"Find the title of the note whose content is about the AI Agent Project."

The plan should say:

"Use query_database to find the title of the note whose content
is about 'AI Agent Project'."

Do not change "content" into "title", "related to", or another
different condition.

The executor is responsible for inspecting the schema and constructing
the SQL query.

13. Calendar questions should use add_calendar_event or
    list_calendar_events.

14. Return only the ordered plan.
"""


# Create the Gemini model.
llm = get_llm()

# Ask the model to return our Pydantic Plan structure.
planner_llm = llm.with_structured_output(Plan)


def create_plan(task: str) -> list[str]:
    """
    Convert a user's task into an ordered list of executable steps.
    """

    response = planner_llm.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=task),
        ]
    )

    steps = response.steps

    # Safety fallback if the model somehow returns an empty plan.
    if not steps:
        return [task]

    # Keep the plan within the project's required 1-6 step range.
    return steps[:6]