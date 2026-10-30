"""

Strategy 1: Chain-of-Thought (CoT) with LangChain.
            Ask the model to write its reasoning step by step 
            BEFORE it gives the final answer.

Flow: question --> prompt --> model --> "Step 1 ... Step 2 ... Answer: X" --> extract X

Key points:
- One straight line, no loops, no tools: a plain LangChain chain is enough (no LangGraph needed).
- LCEL (LangChain Expression Language) joins the parts with the | operator:
      prompt | model | output_parser
- The same question is asked three ways so you can compare:
    (1) Direct        : "just give the answer"         (no reasoning)
    (2) Zero-shot CoT : "let's think step by step"     (reasoning, no example)
    (3) Few-shot CoT  : one worked example first       (reasoning that copies the example's style)
- The final answer is pulled out of the text with a small helper, so the reasoning and the answer are printed separately.

How to run:  uv run 01_chain_of_thought.py

"""

# re: a regular expression finds the "Answer: ..." line in the model's text
import re

# ChatPromptTemplate = a reusable prompt with {placeholders}
# StrOutputParser    = turns the model's message object into a plain string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Two functions there
# get_model(): local Ollama model as a LangChain chat model
# banner(): prints a section title
from common import banner, get_model

# -------------------------------------------------------

# (1) The model and the output parser (shared by all three chains)

# temperature=0 so every run gives the same output and the comparison is fair
model = get_model(temperature=0.0)
parser = StrOutputParser()

# -------------------------------------------------------

# (2) Three prompts for the same question

# (2.1) Direct: no reasoning allowed, only the final number
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer with the final number only. Do not explain."),
    ("human", "{question}"),
])

# (2.2) Zero-shot CoT: the famous trigger phrase "Let's think step by step"
zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You solve word problems. Let's think step by step.
     Write numbered steps, then finish with one last line in the form: Answer: <number>
     """),
    ("human", "{question}"),
])

# (2.3) Few-shot CoT: show ONE solved example so the model copies the format
few_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You solve word problems by reasoning step by step in EXACTLY the same format as the example: "
     "short 'Step N:' lines, then a last line 'Answer: <number>'."),
    # --- worked example (a fake conversation turn) ---
    ("human", "A shop has 15 apples. It sells 6 and then receives 2 boxes of 4 apples. How many apples now?"),
    ("ai",
     "Step 1: Start with 15 apples.\n"
     "Step 2: Sell 6 -> 15 - 6 = 9 apples.\n"
     "Step 3: Receive 2 boxes of 4 -> 2 x 4 = 8 apples.\n"
     "Step 4: Add them -> 9 + 8 = 17 apples.\n"
     "Answer: 17"),
    # --- the real question ---
    ("human", "{question}"),
])

# -------------------------------------------------------

# (3) Build the chains with LCEL:  prompt | model | parser

# Each chain is a "Runnable": call it with .invoke({"question": ...})
direct_chain = direct_prompt | model | parser
zero_shot_cot_chain = zero_shot_cot_prompt | model | parser
few_shot_cot_chain = few_shot_cot_prompt | model | parser

# -------------------------------------------------------

# (4) Run all three on the same question

def main(question: str, expected: str) -> None:
    banner(f"QUESTION\n{question}\n(expected answer: {expected})")

    banner("(1) DIRECT ANSWER (no reasoning)")
    direct = direct_chain.invoke({"question": question})
    print(direct)

    banner("(2) ZERO-SHOT CHAIN-OF-THOUGHT")
    cot = zero_shot_cot_chain.invoke({"question": question})
    print(cot)

    banner("(3) FEW-SHOT CHAIN-OF-THOUGHT")
    few = few_shot_cot_chain.invoke({"question": question})
    print(few)


if __name__ == "__main__":
    # A multi-step problem: easy to get wrong when the model jumps straight to the answer
    main(
        question=(
            "A cafe sells coffee for 45 baht and cake for 60 baht. "
            "Somchai buys 3 coffees and 2 cakes, and pays with a 500 baht note. "
            "The cafe gives a 10 percentage discount on the total. "
            "How much change does he get?"
        ),
        expected="500 - (135 + 120) x 0.9 = 270.5 baht",
    )
