import re
from datetime import datetime


def contains_all(text: str, required_values: list[str]) -> bool:
    text_lower = text.lower()

    return all(
        value.lower() in text_lower
        for value in required_values
    )


def is_valid_datetime(text: str) -> bool:
    matches = re.findall(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        text,
    )

    for value in matches:
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            continue

    return False


def validate_benchmark(
    benchmark: dict,
    state: dict,
) -> tuple[bool, str]:

    if not state:
        return False, "No agent state was returned."

    final_answer = state.get("final_answer", "")

    if not final_answer:
        return False, "The agent returned no final answer."

    if state.get("failed_steps", 0) > 0:
        return False, "One or more agent steps ultimately failed."

    task_id = benchmark["id"]

    # -----------------------------------------------------
    # Python
    # -----------------------------------------------------

    if task_id == "python_01":
        if "100" in final_answer:
            return True, "Final answer contains the expected result: 100."

        return False, "Expected Python result 100 was not found."

    if task_id == "python_02":
        if "355687428096000" in final_answer.replace(",", ""):
            return True, "Final answer contains the expected factorial result."

        return False, "Expected factorial result was not found."

    if task_id == "python_03":
        required = [
            "0",
            "1",
            "2",
            "3",
            "5",
            "8",
            "13",
            "21",
            "34",
        ]

        if contains_all(final_answer, required):
            return True, "Final answer contains the first 10 Fibonacci numbers."

        return False, "Expected Fibonacci sequence was not found."

    # -----------------------------------------------------
    # Date/time
    # -----------------------------------------------------

    if task_id == "datetime_01":
        if is_valid_datetime(final_answer):
            return True, "Final answer contains a valid date and time."

        return False, "No valid YYYY-MM-DD HH:MM:SS timestamp found."

    if task_id == "datetime_python_01":
        if "9" in final_answer:
            return True, "Final answer contains the expected calculation result 9."

        return False, "Expected calculation result 9 was not found."

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    if task_id == "database_01":
        required = [
            "AI Agent Project",
            "Thesis",
            "Research",
        ]

        if contains_all(final_answer, required):
            return True, "Final answer contains all expected note titles."

        return False, "One or more expected note titles are missing."

    if task_id == "database_02":
        if "AI Agent Project" in final_answer:
            return True, "Expected note title was found."

        return False, "Expected note title 'AI Agent Project' was not found."

    if task_id == "database_03":
        if "AI Agent Project" in final_answer:
            return True, "Expected recovered note title was found."

        return False, "Expected recovered note title was not found."

    # -----------------------------------------------------
    # Web
    # -----------------------------------------------------

    if task_id == "web_01":
        if "Guido van Rossum" in final_answer:
            return True, "Expected Python creator was found."

        return False, "Expected Python creator 'Guido van Rossum' was not found."

    # -----------------------------------------------------
    # Calendar
    # -----------------------------------------------------

    if task_id == "calendar_01":
        required = [
            "Team Meeting",
            "14:00",
        ]

        if not contains_all(final_answer, required):
            return False, "Expected calendar event title or time was not found."

        date_variants = [
            "2040-06-20",
            "June 20, 2040",
            "June 20 2040",
            "20 June 2040",
            "20th June 2040",
        ]

        if any(
            variant.lower() in final_answer.lower()
            for variant in date_variants
        ):
            return True, "Final answer contains the expected calendar event details."

        return False, "Expected calendar event date was not found."


    if task_id == "calendar_02":
        messages = state.get("messages", [])
        tool_used = False
        tool_evidence = False

        for message in messages:
            # Check whether the required tool was called.
            tool_calls = getattr(message, "tool_calls", [])

            for tool_call in tool_calls:
                if tool_call.get("name") == "list_calendar_events":
                    tool_used = True

            # ToolMessage contains the actual tool output.
            message_type = type(message).__name__
            content = getattr(message, "content", "")

            if isinstance(content, str) and message_type == "ToolMessage":
                if (
                    "AI Agent Demo" in content
                    and "Project Demo" in content
                    and "Automated Test Event" in content
                ):
                    tool_evidence = True

        if tool_used and tool_evidence:
            return (
                True,
                "list_calendar_events was called and returned actual calendar-event records.",
            )

        return (
            False,
            "Could not verify the calendar tool call and its actual output.",
        )
    # -----------------------------------------------------
    # Multi-step
    # -----------------------------------------------------

    if task_id == "multi_01":
        if "9" in final_answer:
            return True, "Final answer contains the expected calculation result 9."

        return False, "Expected calculation result 9 was not found."

    # -----------------------------------------------------
    # Unknown
    # -----------------------------------------------------

    return (
        False,
        f"No validator has been defined for benchmark '{task_id}'.",
    )
