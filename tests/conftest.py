
import os
import sqlite3
from pathlib import Path

import pytest


# Tests use a separate database, never the main agent database.
TEST_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "test_agent.db"
)

os.environ["AGENT_DB_PATH"] = str(TEST_DB_PATH)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Create clean sample data for the test session."""
    TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove only the dedicated test database from a previous run.
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    with sqlite3.connect(TEST_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.executemany(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            [
                (
                    "AI Agent Project",
                    "Build a LangGraph autonomous AI agent with multiple tools.",
                ),
                (
                    "Thesis",
                    "Fine-tuning open-source language models for medical question answering.",
                ),
                (
                    "Research",
                    "Explore LLM agents, tool calling, planning, verification, and evaluation.",
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO calendar_events
                (title, event_date, event_time, description)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "AI Agent Demo",
                    "2040-01-01",
                    "09:00",
                    "Sample event for tests",
                ),
                (
                    "Project Demo",
                    "2040-01-02",
                    "10:00",
                    "Another sample event for tests",
                ),
            ],
        )

    yield

    # Leave the dedicated test database in place for now.
    # It will be reset at the beginning of the next test session.