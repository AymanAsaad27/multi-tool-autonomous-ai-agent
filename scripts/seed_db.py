import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("AGENT_DB_PATH", "data/agent.db"))

SAMPLE_NOTES = [
    (
        "AI Agent Project",
        "Build a LangGraph autonomous AI agent with multiple tools.",
    ),
    (
        "Thesis",
        "Fine-tuning open-source large language models for medical question answering.",
    ),
    (
        "Research",
        "Explore LLM agents, tool calling, planning, verification, and evaluation.",
    ),
]


def create_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for title, content in SAMPLE_NOTES:
            cursor.execute(
                """
                INSERT INTO notes (title, content)
                SELECT ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM notes WHERE title = ?
                )
                """,
                (title, content, title),
            )

    print(f"Database initialized successfully: {DB_PATH}")


if __name__ == "__main__":
    create_database()
