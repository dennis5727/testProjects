# Reasoning Strategies (uv + LangChain + LangGraph + Ollama)

Three small scripts that show the main LLM reasoning strategies:
**Chain-of-Thought, ReAct and Reflexion**.
Each one uses the framework that fits it best:

- **LangChain** for a straight line of steps (prompt → model → parser).
- **LangGraph** when the reasoning needs **state and loops**.

The model runs **locally with Ollama**, so no API key is needed.

## 1. Prerequisites

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama from https://ollama.com, then download the model
ollama pull qwen2.5:7b    # about 4.7 GB
```

## 2. Setup with uv

```bash
uv sync                 # creates .venv and installs from uv.lock
```

`.env` is optional. Without it, `common.py` uses these defaults:

```
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

```bash
ollama pull qwen3:8b
OLLAMA_MODEL=qwen3:8b OLLAMA_REASONING=false uv run 01_chain_of_thought.py
```

This project was created with:

```bash
uv init --bare --python 3.12
uv add langchain langchain-core langchain-ollama langgraph python-dotenv
```

## 3. Run the examples

```bash
uv run 01_chain_of_thought.py
uv run 02_react.py
uv run 03_reflexion.py
```

To try your own input, edit it at the bottom (or the top) of each script:

| Script | What to change |
|---|---|
| `01_chain_of_thought.py` | The `question` and `expected` in `main(...)` |
| `02_react.py` | The question in `main(...)`; add cities to `CITY_FACTS` or add a new `@tool` |
| `03_reflexion.py` | `TASK`, `TESTS` and `MAX_ATTEMPTS` at the top |

`common.py` holds the shared setup used by every script: `get_model(temperature)` (the Ollama chat model) and `banner()` (prints a section title).

## 4. The strategies

| Script | Strategy | Framework | How the reasoning flows | Use it when |
|---|---|---|---|---|
| `01_chain_of_thought.py` | Chain-of-Thought | LangChain (LCEL `prompt \| model \| parser`) | One line: think step by step, then answer | Multi-step maths / logic where jumping to the answer causes mistakes |
| `02_react.py` | ReAct | LangGraph (loop: agent ⇄ tools) | Thought → Action (tool) → Observation → … → Answer | The model needs facts or exact calculations it does not have |
| `03_reflexion.py` | Reflexion | LangGraph (loop: coder → evaluator → reflect) | Try, get feedback, write a lesson, try again with the lessons | There is a clear success check (tests, rules, a judge) and retries are allowed |

### What you should see

- **CoT:** the direct answer is wrong (`255` on our machine, the bill before the discount; other setups may give a different wrong number); both CoT versions show their steps and get `270.5`.
- **ReAct:** the model looks up both cities, calls the calculator, and then answers. Tool calls in the same step run in parallel.
- **Reflexion:** attempt 1 fails some tests, the reflection names the failing inputs, and a later attempt passes all 7 tests. The final printout lists the lessons kept in memory.