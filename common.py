"""Shared setup for every reasoning example: the local Ollama chat model for LangChain / LangGraph."""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# (1) Load settings from .env (falls back to the defaults below)
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# (2) Thinking models such as qwen3 reason privately before answering.
#     OLLAMA_REASONING=false turns that off, so "direct" really means direct.
#     Leave it unset for models without a thinking mode (such as qwen2.5).
_reasoning = os.getenv("OLLAMA_REASONING", "").strip().lower()
OLLAMA_REASONING = {"true": True, "false": False}.get(_reasoning)

# -------------------------------------------------------

def get_model(temperature: float = 0.0) -> ChatOllama:
    """Return the local Ollama model as a LangChain chat model.

    temperature = 0.0 -> the same answer every run (good for reasoning and judging)
    temperature > 0.0 -> more varied answers (good for brainstorming many ideas)
    """
    return ChatOllama(model=OLLAMA_MODEL, 
                      base_url=OLLAMA_BASE_URL, 
                      temperature=temperature,
                      reasoning=OLLAMA_REASONING)


def banner(title: str) -> None:
    """Print a section header so the console output is easy to follow."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
