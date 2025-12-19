"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from mlflow.entities import SpanType
from typing_extensions import TypedDict

from agent.utils.config import (
    init_models,
    setup_mlflow_tracing,
    system_prompt,
)
from agent.utils.rag import build_prompt

# Set NO_PROXY to avoid proxy for localhost connections (important for local MCP server access)
#setup_no_proxy()
#os.environ["https_proxy"] = ""
#os.environ["http_proxy"] = ""

# load env values (LLM api key etc.)

load_dotenv()

@dataclass
class ContextSchema:
    model_provider: str = "openai"
    system_message: str = system_prompt()

class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    test_output: str = "example"
    input_messages: list[AnyMessage] = ["How to run Docker?"]
    #input_messages: str
    system_prompt: str = system_prompt()
    messages: list[AnyMessage]

@mlflow.trace(span_type=SpanType.TOOL)
def rag_search(state:State):
    """Node to search in the rag base to enhance the context for the graph."""
    input_messages = build_prompt(state["input_messages"][-1])

    return {"input_messages": input_messages}

def call_model(state: State, runtime: Runtime[ContextSchema]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    MODELS = init_models()
    #model = MODELS[runtime.context.model_provider]
    model = MODELS["openai"]
    #input_messages = build_prompt(state["input_messages"][-1])
    #input_messages = build_prompt("How to run Docker?")
    input_messages = state["input_messages"]
    response = model.invoke(input_messages)

    return {
        "test_output": "test output from call_model. ",
        "input_messages": state["input_messages"],
        "system_prompt": system_prompt(),
        "messages": [response],
    }

# Define the graph
graph = (
    StateGraph(State, context_schema=ContextSchema)
    .add_edge("__start__", "rag_search")
    .add_node(rag_search)
    .add_node(call_model)
    .add_edge("rag_search","call_model")
    .compile(name="New Graph")
)

@mlflow.trace(span_type=SpanType.AGENT)
def run_graph(query: str, langfuse_handler=None):
    """Run the graph with the given query."""
    result = graph.invoke(
       {"input_messages": [query]},
       #config={"callbacks": [langfuse_handler]}
   )

    for m in result["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    # Example of running the graph directly

    # Optionally set up tracing (langfuse or mlflow)
    # langfuse_handler = setup_langfuse()
    setup_mlflow_tracing()
    
    query = input("Enter your question: ")
    
    run_graph(query)
