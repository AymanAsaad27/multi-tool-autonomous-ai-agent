from agent.tools import (
    get_current_datetime,
    web_search,
    run_python,
    query_database,
    add_calendar_event,
    list_calendar_events,
)


def test_get_current_datetime():
    result = get_current_datetime.invoke({})

    assert result
    assert len(result) == 19
    assert result[4] == "-"
    assert result[7] == "-"
    assert result[10] == " "


def test_run_python():
    result = run_python.invoke({
        "code": "print(25 * 4)"
    })

    assert result == "100"


def test_query_database():
    result = query_database.invoke({
        "query": "SELECT title FROM notes"
    })

    assert "AI Agent Project" in result
    assert "Thesis" in result
    assert "Research" in result


def test_query_database_is_read_only():
    result = query_database.invoke({
        "query": "DROP TABLE notes"
    })

    assert "Only read-only SQL queries are allowed." in result


def test_add_calendar_event():
    result = add_calendar_event.invoke({
        "title": "Pytest Calendar Test",
        "event_date": "2040-01-01",
        "event_time": "12:00",
        "description": "Created by pytest",
    })

    assert "Calendar event added successfully" in result


def test_list_calendar_events():
    result = list_calendar_events.invoke({})

    assert "AI Agent Demo" in result
    assert "Project Demo" in result