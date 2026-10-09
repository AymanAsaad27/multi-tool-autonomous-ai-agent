import argparse
import os
import sqlite3
import time
from pathlib import Path

BENCHMARK_DB_PATH = Path("data/benchmark_agent.db")
os.environ["AGENT_DB_PATH"] = str(BENCHMARK_DB_PATH)

from agent.run import run_agent
from evals.tasks import BENCHMARK_TASKS
from evals.validators import validate_benchmark

def initialize_benchmark_database():
    """Reset the isolated benchmark database with predictable sample data."""
    BENCHMARK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(BENCHMARK_DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS notes")
        cursor.execute("DROP TABLE IF EXISTS calendar_events")

        cursor.execute("""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.executemany(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            [
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
            ],
        )

        cursor.executemany(
            """
            INSERT INTO calendar_events
                (title, event_date, event_time, description)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "AI Agent Demo",
                    "2026-10-10",
                    "15:00",
                    "Demonstrate the autonomous AI agent",
                ),
                (
                    "Project Demo",
                    "2026-11-10",
                    "15:00",
                    "",
                ),
                (
                    "Automated Test Event",
                    "2030-01-01",
                    "10:00",
                    "Created by automated tests",
                ),
            ],
        )

        conn.commit()
    finally:
        conn.close()


def run_benchmark(task_id=None):
    """
    Run all benchmark tasks, or one specific task when task_id is provided.
    """
    initialize_benchmark_database()
    if task_id:
        selected_tasks = [
            benchmark
            for benchmark in BENCHMARK_TASKS
            if benchmark["id"] == task_id
        ]

        if not selected_tasks:
            available_ids = ", ".join(
                benchmark["id"]
                for benchmark in BENCHMARK_TASKS
            )

            raise ValueError(
                f"Unknown benchmark ID: {task_id}\n"
                f"Available IDs: {available_ids}"
            )
    else:
        selected_tasks = BENCHMARK_TASKS

    results = []

    for index, benchmark in enumerate(selected_tasks, start=1):

        print("\n" + "=" * 70)
        print(f"BENCHMARK {index}/{len(selected_tasks)}")
        print(f"ID: {benchmark['id']}")
        print(f"CATEGORY: {benchmark['category']}")
        print(f"TASK: {benchmark['task']}")
        print("=" * 70)

        start_time = time.perf_counter()

        state = run_agent(benchmark["task"])

        elapsed_time = time.perf_counter() - start_time

        validation_success, validation_reason = (
            validate_benchmark(
                benchmark,
                state,
            )
        )

        result = {
            "id": benchmark["id"],
            "category": benchmark["category"],
            "task": benchmark["task"],
            "success": validation_success,
            "validation_reason": validation_reason,
            "tool_calls": (
                state.get("total_tool_calls", 0)
                if state
                else 0
            ),
            "failed_steps": (
                state.get("failed_steps", 0)
                if state
                else 0
            ),
            "retries": (
                state.get("total_retries", 0)
                if state
                else 0
            ),
            "latency_seconds": round(
                elapsed_time,
                2,
            ),
            "final_answer": (
                state.get("final_answer", "")
                if state
                else ""
            ),
        }

        results.append(result)

        print("\nBENCHMARK RESULT:")
        print(f"Success: {result['success']}")
        print(f"Validation: {result['validation_reason']}")
        print(f"Tool calls: {result['tool_calls']}")
        print(f"Failed steps: {result['failed_steps']}")
        print(f"Retries: {result['retries']}")
        print(f"Latency: {result['latency_seconds']} seconds")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the autonomous AI agent benchmark."
    )

    parser.add_argument(
        "--id",
        dest="task_id",
        help="Run only the benchmark with this ID.",
    )

    args = parser.parse_args()

    results = run_benchmark(args.task_id)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(f"Tasks completed: {len(results)}")

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    success_rate = (
        successful / len(results) * 100
        if results
        else 0
    )

    print(
        f"Successful tasks: "
        f"{successful}/{len(results)}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.1f}%"
    )


if __name__ == "__main__":
    main()