import sys

from agent.graph import build_graph
from agent.state import initial_state
from agent.text import text_of


def run_agent(question: str):
    """
    Run the autonomous agent for a given task.

    Returns the final agent state after the graph finishes.
    """

    app = build_graph()

    final_state = None

    for event in app.stream(
        initial_state(question),
        config={"recursion_limit": 80},
        stream_mode="updates",
    ):

        for node, update in event.items():

            # Keep the most recent state updates.
            if final_state is None:
                final_state = initial_state(question)


            for key, value in update.items():
                if key == "messages":
                    from langgraph.graph.message import add_messages

                    final_state["messages"] = add_messages(
                        final_state.get("messages", []),
                        value,
                    )
                else:
                    final_state[key] = value

            # -----------------------------------------
            # Planner
            # -----------------------------------------

            if node == "planner":

                print("\nPLAN:")

                for i, step in enumerate(
                    update["plan"],
                    start=1,
                ):
                    print(f"  {i}. {step}")

            # -----------------------------------------
            # Begin step
            # -----------------------------------------

            elif node == "begin_step":

                print(
                    "\n"
                    + "-" * 60
                )

                print("BEGINNING STEP")

            # -----------------------------------------
            # Executor
            # -----------------------------------------

            elif node == "executor":

                message = update["messages"][-1]

                tool_calls = getattr(
                    message,
                    "tool_calls",
                    [],
                )

                if tool_calls:

                    print("\nEXECUTOR → TOOL")

                    for tool_call in tool_calls:

                        print(
                            f"  -> {tool_call['name']}"
                            f"({tool_call['args']})"
                        )

                else:

                    content = text_of(message)

                    if content:
                        print("\nEXECUTOR:")
                        print(content)

            # -----------------------------------------
            # Tools
            # -----------------------------------------

            elif node == "tools":

                print("\nTOOL RESULT:")

                for message in update["messages"]:

                    result = text_of(message)

                    print(
                        f"  <- {result[:500]}"
                    )

            # -----------------------------------------
            # Verifier
            # -----------------------------------------

            elif node == "verify":

                results = update.get("results")

                if results:
                    print("\nVERIFIER:")
                    print(
                        f"  ✓ Step result: "
                        f"{results[-1]}"
                    )

                else:
                    print(
                        "\nVERIFIER: "
                        "Step failed — retrying..."
                    )

            # -----------------------------------------
            # Synthesizer
            # -----------------------------------------

            elif node == "synthesize":

                print(
                    "\n"
                    + "=" * 60
                )

                print("FINAL ANSWER:")

                print(
                    update["final_answer"]
                )

                print(
                    "=" * 60
                )

    return final_state


def main():

    # Get the user's task from the command line.
    question = " ".join(sys.argv[1:])

    if not question:
        question = input("Task: ").strip()

    if not question:
        print("No task provided.")
        return

    print("\n" + "=" * 60)
    print("MULTI-TOOL AUTONOMOUS AI AGENT")
    print("=" * 60)

    print(f"\nTASK:\n{question}")

    print("\nStarting agent...\n")

    run_agent(question)


if __name__ == "__main__":
    main()