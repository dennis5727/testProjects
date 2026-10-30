"""

Strategy 4: Reflexion with LangGraph.
            Try -> get feedback -> write a lesson ("reflection") -> try again WITH the lessons in memory.
            The model improves by learning from its own mistakes, without any training.

Task: write a Python function. The feedback comes from running unit tests on it
      (the same "programming" setting used in the Reflexion paper).

Key points:
- Three roles, three nodes:
    (1) coder     : writes the code (the Reflexion paper calls this role the "actor").
                    It reads the task, its last attempt + test results (short-term memory)
                    AND every lesson so far (long-term memory).
    (2) evaluator : runs the unit tests and reports which ones failed (an objective signal, no LLM).
    (3) reflect   : the model explains WHY it failed and what to do next time (one short lesson).
- memory (the list of reflections) uses Annotated[list, operator.add], so each lesson is appended, never overwritten.
- The conditional edge stops when all tests pass or after MAX_ATTEMPTS.
- The evaluator could also be an LLM judge (e.g. for essays); tests are used here because they cannot be fooled.
- Note: the generated code is run in a separate Python process with a timeout. That is fine for a classroom demo,
  but never run untrusted model code on a machine with important data.

How to run:  uv run 03_reflexion.py

"""

import operator
# re: pulls the ```python ... ``` code block out of the model's reply
import re
# subprocess + sys + tempfile: run the generated code in a separate Python process
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, get_model

# -------------------------------------------------------

# (1) The task and its unit tests

MAX_ATTEMPTS = 4

TASK = (
    """
    Write a Python function format_name(full_name: str) -> str that returns the name
    in the form "Last, First". Example: "Ada Lovelace" -> "Lovelace, Ada".
    """
)

# The tests check edge cases the short task description does not spell out.
# The coder only learns about them from the feedback, which is exactly what Reflexion is for.
TESTS = """
cases = [
    ("Ada Lovelace", "Lovelace, Ada"),
    ("  alan   turing ", "Turing, Alan"),                    # extra spaces + lower case
    ("ada LOVELACE", "Lovelace, Ada"),                       # each word Capitalised
    ("Grace Brewster Hopper", "Hopper, Grace Brewster"),     # middle names stay after the first name
    ("Plato", "Plato"),                                      # a single name has no comma
]
failed = 0
for text, expected in cases:
    try:
        got = format_name(text)
    except Exception as error:
        got = f"{type(error).__name__}: {error}"
    if got != expected:
        failed += 1
        print(f"FAIL format_name({text!r}) -> {got!r}, expected {expected!r}")

for bad in ["", "   "]:                                     # empty input must be rejected
    try:
        got = format_name(bad)
        failed += 1
        print(f"FAIL format_name({bad!r}) -> {got!r}, expected a ValueError")
    except ValueError:
        pass
    except Exception as error:
        failed += 1
        print(f"FAIL format_name({bad!r}) raised {type(error).__name__}, expected a ValueError")

total = len(cases) + 2
print(f"{total - failed}/{total} tests passed")
"""

# -------------------------------------------------------

# (2) The graph state

class ReflexionState(TypedDict):
    task: str
    code: str
    test_report: str
    passed: bool
    attempt: int
    memory: Annotated[list[str], operator.add]   # the reflections, appended one by one

# -------------------------------------------------------

# (3) Helpers

model = get_model(temperature=0.0)

def extract_code(text: str) -> str:
    """Return the first ```python block (or the whole reply if there is none)."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()

def run_tests(code: str) -> tuple[bool, str]:
    """Run the code + the tests in a separate Python process."""
    with tempfile.TemporaryDirectory() as folder:
        script = Path(folder) / "attempt.py"
        script.write_text(code + "\n\n" + TESTS)
        try:
            # -I = isolated mode: ignore environment variables and the user's site-packages
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False, "The code took too long (infinite loop?)."

    report = (result.stdout + result.stderr).strip()
    passed = result.returncode == 0 and "FAIL" not in report
    return passed, report

# -------------------------------------------------------

# (4) The nodes

def coder(state: ReflexionState) -> dict:
    """Write (or rewrite) the code, using every lesson learned so far."""
    attempt = state["attempt"] + 1
    lessons = "\n".join(f"- {lesson}" for lesson in state["memory"]) or "(none yet)"

    # After the first try, also show the last attempt and its test results (short-term memory)
    previous = ""
    if state["code"]:
        previous = (
            f"Your previous attempt:\n```python\n{state['code']}\n```\n"
            f"Its test results:\n{state['test_report']}\n\n"
        )

    prompt = (
        f"{state['task']}\n"
        f"{previous}"
        f"Lessons from your previous attempts (follow ALL of them):\n{lessons}\n\n"
        f"Reply with only the complete function in one ```python code block. "
        f"Use only the standard library."
    )
    code = extract_code(model.invoke(prompt).content)

    banner(f"ATTEMPT {attempt}: coder wrote")
    print(code)
    return {"code": code, "attempt": attempt}


def evaluator(state: ReflexionState) -> dict:
    """Run the unit tests: an objective pass/fail signal."""
    passed, report = run_tests(state["code"])
    print(f"\n--- evaluator ---\n{report}")
    return {"passed": passed, "test_report": report}


def reflect(state: ReflexionState) -> dict:
    """Look at the failures and write ONE lesson for the next attempt."""
    prompt = (
        f"You wrote this code for the task below, but some tests failed.\n\n"
        f"Task:{state['task']}\n"
        f"Code:\n```python\n{state['code']}\n```\n\n"
        f"Test results:\n{state['test_report']}\n\n"
        f"In 1-3 short sentences: what did you get wrong, and what exactly must the next version do differently? "
        f"Mention the specific inputs that failed. Do not write code."
    )
    lesson = model.invoke(prompt).content.strip()
    print(f"\n--- reflection (added to memory) ---\n{lesson}")
    return {"memory": [lesson]}

# -------------------------------------------------------

# (5) The conditional edge: stop or reflect and try again?

def should_continue(state: ReflexionState) -> Literal["reflect", "__end__"]:
    if state["passed"] or state["attempt"] >= MAX_ATTEMPTS:
        return END
    return "reflect"

# -------------------------------------------------------

# (6) Build the graph (the Reflexion loop)

#   START --> coder --> evaluator --(passed, or no tries left)--> END
#               ^           |
#               |    (failed, tries left)
#               |           v
#               +------- reflect   (writes one lesson into memory)

# The state is ReflexionState: the task, the latest code, its test report, the attempt number
# and the memory of lessons, shared by every node.
builder = StateGraph(ReflexionState)

# Nodes: give each function a name so the edges can refer to it
builder.add_node("coder", coder)           # the model writes the code, reading every lesson in memory
builder.add_node("evaluator", evaluator)   # Python runs the unit tests (no model, so it cannot be fooled)
builder.add_node("reflect", reflect)       # the model turns the failed tests into one lesson

# Normal edges: always run in this order
builder.add_edge(START, "coder")           # the first attempt starts with an empty memory
builder.add_edge("coder", "evaluator")     # every new version of the code is tested

# Conditional edge: after the evaluator, should_continue picks the next node
#   - all tests passed, or MAX_ATTEMPTS reached -> END
#   - some tests failed and tries are left       -> "reflect"
builder.add_conditional_edges("evaluator", should_continue)

# Normal edge: after reflecting, always try again. This edge makes the loop,
# and the new lesson is already in memory when the coder runs again.
builder.add_edge("reflect", "coder")

# compile() checks the graph and turns it into something we can run with invoke()
graph = builder.compile()

# -------------------------------------------------------

# (7) Run it

def main() -> None:
    final = graph.invoke({
        "task": TASK,
        "code": "",
        "test_report": "",
        "passed": False,
        "attempt": 0,
        "memory": [],
    })

    banner("RESULT")
    status = "PASSED" if final["passed"] else "still failing"

    # Printing status
    print(f"{status} after {final['attempt']} attempt(s).")

    # Printing lesson
    print("\nMemory (lessons learned):")
    for i, lesson in enumerate(final["memory"], start=1):
        print(f"  {i}. {lesson}")


if __name__ == "__main__":
    main()
