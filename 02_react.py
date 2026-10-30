"""

Strategy 3: ReAct (Reason + Act) with LangGraph.
            The model alternates between THINKING and ACTING (calling a tool),
            reads the tool's result (the observation), and thinks again until it can answer.

Key points:
- Two tools (local):
    (1) lookup_city     : a tiny "knowledge base" of city facts
    (2) calculator      : one arithmetic step, a op b (the model is bad at exact maths, so it should not do it in its head)

- model.bind_tools(tools) tells the model which tools exist and what arguments they take.

- The loop:
    agent node  ->  tools_condition (did the model ask for a tool?)  ->  ToolNode (runs it)  ->  back to agent

- graph.invoke() runs the loop; then every message is printed: Thought -> Action -> Observation -> ... -> Final Answer.

- recursion_limit stops the loop if the model never finishes.

How to run:  uv run 02_react.py

"""

# SystemMessage = instructions, HumanMessage = the user, AIMessage = the model, ToolMessage = a tool's result
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# @tool turns a normal Python function into a tool the model can call (name, docstring and arguments are sent to the model)
from langchain_core.tools import tool

# MessagesState = a ready-made state with one field: "messages" (new messages are appended)
from langgraph.graph import END, START, MessagesState, StateGraph

# ToolNode       = a node that runs whatever tool calls the last AIMessage asked for
# tools_condition = a router: go to "tools" if the last AIMessage has tool calls, otherwise END
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, get_model

# -------------------------------------------------------

# (1) The tools

# Data for the example
CITY_FACTS = {
    "bangkok": {"country": "Thailand", "population": 10_500_000, "area_km2": 1_569},
    "tokyo": {"country": "Japan", "population": 14_000_000, "area_km2": 2_194},
    "singapore": {"country": "Singapore", "population": 5_900_000, "area_km2": 735},
    "chiang mai": {"country": "Thailand", "population": 1_200_000, "area_km2": 20_107},
}

@tool
def lookup_city(city: str) -> str:
    """Look up facts about a city: country, population and area in km2."""
    facts = CITY_FACTS.get(city.strip().lower())
    if facts is None:
        return f"No data for '{city}'. Known cities: {', '.join(CITY_FACTS)}."
    return f"{city}: {facts}"


@tool
def calculator(a: float, op: str, b: float) -> str:
    """Calculate a op b, where op is +, -, * or /. Example: a=10500000, op="/", b=1569."""
    if op == "+": result = a + b
    elif op == "-": result = a - b
    elif op == "*": result = a * b
    elif op == "/": result = a / b
    else: return f"Error: unknown operator {op}"
    return f"{a} {op} {b} = {round(result, 4)}"

# -------------------------------------------------------

# (2) The model, with the tools attached

SYSTEM_PROMPT = (
    """
    You are a helpful assistant that solves questions step by step.
    Before each action, briefly say what you are thinking.
    Use lookup_city to get facts. Never guess numbers.
    Use calculator for every calculation. Never do arithmetic in your head.
    When you have everything you need, give the final answer in one or two sentences.
    """
)

tools = [lookup_city, calculator]
model_with_tools = get_model(temperature=0.0).bind_tools(tools)

# -------------------------------------------------------

# (3) The nodes

# Agent node
def agent(state: MessagesState) -> dict:
    """Reason: read the conversation so far, then either call a tool or answer."""
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}   # appended to the history by MessagesState

# Tool node
tool_node = ToolNode(tools)           # Act: run the requested tool(s) and return ToolMessage(s)

# -------------------------------------------------------

# (4) Build the graph (the ReAct loop)

#   START --> agent --(asks for a tool)--> tools
#               ^                            |
#               +-------- observation -------+
#             agent ----(no tool call)----> END

# The state is MessagesState: one list of messages shared by every node.
# Each node returns new messages, and they are added to the end of that list.
builder = StateGraph(MessagesState)

# Nodes: give each function a name so the edges can refer to it
builder.add_node("agent", agent)       # Thought: the model decides what to do next
builder.add_node("tools", tool_node)   # Action + Observation: run the tool(s), return the results

# Edges: the order the nodes run in
builder.add_edge(START, "agent")       # The run always begins with the model reading the question

# Conditional edge: after the agent, tools_condition looks at the model's last message
#   - it asked for a tool (tool_calls)  -> go to "tools"
#   - it gave a plain answer            -> go to END (the run is finished)
builder.add_conditional_edges("agent", tools_condition)

# Normal edge: after the tools run, always go back to the agent,
# so the model can read the observation and think again. This edge makes the loop.
builder.add_edge("tools", "agent")

# compile() checks the graph and turns it into something we can run with invoke() or stream()
graph = builder.compile()

# -------------------------------------------------------

# (5) Run it and print each step

# Function for printing message in the state out
def show(message) -> None:
    """Print one message in ReAct style."""
    if isinstance(message, AIMessage):
        if message.content:
            label = "Thought" if message.tool_calls else "Final Answer"
            print(f"\n[{label}] {message.content.strip()}")
        for call in message.tool_calls:
            print(f"[Action] {call['name']}({call['args']})")
    elif isinstance(message, ToolMessage):
        print(f"[Observation] {message.content}")

def main(question: str) -> None:
    banner(f"REACT\n{question}")

    # Run the whole ReAct loop. The starting state is one message: the user's question.
    # recursion_limit stops the loop if the model never stops calling tools.
    result = graph.invoke({"messages": [HumanMessage(question)]},
                          config={"recursion_limit": 12})

    # result["messages"] holds every step in order: Thought -> Action -> Observation -> ... -> Final Answer
    for message in result["messages"]:
        show(message)


if __name__ == "__main__":
    main("Which city is more crowded (people per km2), Bangkok or Tokyo, and by how many people per km2?")

    # Other questions to try:
    # main("How many Singapores would fit inside the area of Chiang Mai?")