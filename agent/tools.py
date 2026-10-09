from datetime import datetime
from pathlib import Path
import sqlite3
import os
from langchain_core.tools import tool


DB_PATH = Path(
    os.environ.get("AGENT_DB_PATH", "data/agent.db")
)

@tool
def get_current_datetime() -> str:
    """
    Get the current local date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str) -> str:
    """
    Search the web for information related to the given query.
    """
    from ddgs import DDGS

    results = DDGS().text(query, max_results=5)

    if not results:
        return "No search results found."

    output = []

    for i, result in enumerate(results, start=1):
        title = result.get("title", "")
        body = result.get("body", "")
        url = result.get("href", "")

        output.append(
            f"{i}. {title}\n"
            f"   {body}\n"
            f"   URL: {url}"
        )

    return "\n\n".join(output)



@tool
def run_python(code: str) -> str:
    """
    Execute a small Python program and return its output or result.
    """
    try:
        import ast
        import io
        from contextlib import redirect_stdout

        namespace = {}
        output = io.StringIO()

        with redirect_stdout(output):
            tree = ast.parse(code, mode="exec")

            # Evaluate the final expression, such as `2026 - 2017`.
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

                exec(
                    compile(
                        ast.Module(
                            body=tree.body,
                            type_ignores=[],
                        ),
                        "<agent>",
                        "exec",
                    ),
                    {},
                    namespace,
                )

                result = eval(
                    compile(
                        ast.Expression(last_expr.value),
                        "<agent>",
                        "eval",
                    ),
                    {},
                    namespace,
                )

            else:
                exec(code, {}, namespace)
                result = None

                # Return the value of a final simple variable assignment.
                if tree.body:
                    last_stmt = tree.body[-1]
                    target = None

                    if isinstance(last_stmt, ast.Assign):
                        if last_stmt.targets:
                            target = last_stmt.targets[-1]
                    elif isinstance(last_stmt, (ast.AnnAssign, ast.AugAssign)):
                        target = last_stmt.target

                    if isinstance(target, ast.Name):
                        result = namespace.get(target.id)

        printed = output.getvalue().strip()

        if printed:
            return printed

        if result is not None:
            return str(result)

        return "Python executed successfully with no output."

    except Exception as e:
        return f"Python execution error: {type(e).__name__}: {e}"
@tool
def query_database(query: str) -> str:
    """
    Execute a read-only SQL query against the SQLite database.
    """

    dangerous_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
    ]

    upper_query = query.upper()

    if any(keyword in upper_query for keyword in dangerous_keywords):
        return "Only read-only SQL queries are allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        column_names = [
            description[0]
            for description in cursor.description
        ]

        # If the query returned no rows, provide useful recovery
        # information to the agent.
        if not rows:
            try:
                # Extract the table name from a normal SELECT query.
                table_name = (
                    query
                    .split("FROM")[1]
                    .split()[0]
                    .strip('"`[]')
                )

                schema_cursor = conn.cursor()
                schema_cursor.execute(
                    f"PRAGMA table_info({table_name})"
                )

                schema_rows = schema_cursor.fetchall()

                columns = [
                    row[1]
                    for row in schema_rows
                ]

                conn.close()

                return (
                    "No results found for this query.\n"
                    f"Table '{table_name}' columns: "
                    f"{', '.join(columns)}.\n"
                    "The query may have searched the wrong column "
                    "or condition. Use the discovered columns to "
                    "construct a different query."
                )

            except Exception:
                conn.close()

                return (
                    "No results found for this query. "
                    "The query may have searched the wrong column "
                    "or condition. Try another query using the "
                    "available table columns."
                )

        conn.close()

        output = [
            ", ".join(column_names)
        ]

        for row in rows:
            output.append(
                ", ".join(str(value) for value in row)
            )

        return "\n".join(output)

    except Exception as e:
        return f"Database error: {type(e).__name__}: {e}"


@tool
def add_calendar_event(
    title: str,
    event_date: str,
    event_time: str = "",
    description: str = "",
) -> str:
    """
    Add an event to the local SQLite calendar.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO calendar_events
            (title, event_date, event_time, description)
            VALUES (?, ?, ?, ?)
            """,
            (
                title,
                event_date,
                event_time,
                description,
            ),
        )

        conn.commit()

        event_id = cursor.lastrowid

        conn.close()

        return (
            f"Calendar event added successfully. "
            f"Event ID: {event_id}"
        )

    except Exception as e:
        return f"Calendar error: {type(e).__name__}: {e}"


@tool
def list_calendar_events() -> str:
    """
    List all calendar events stored in the local SQLite database.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, title, event_date, event_time, description
            FROM calendar_events
            ORDER BY event_date, event_time
            """
        )

        rows = cursor.fetchall()

        conn.close()

        if not rows:
            return "No calendar events found."

        output = []

        for row in rows:
            event_id, title, date, time, description = row

            output.append(
                f"ID: {event_id}\n"
                f"Title: {title}\n"
                f"Date: {date}\n"
                f"Time: {time or 'Not specified'}\n"
                f"Description: {description or 'None'}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Calendar error: {type(e).__name__}: {e}"


TOOLS = [
    get_current_datetime,
    web_search,
    run_python,
    query_database,
    add_calendar_event,
    list_calendar_events,
]