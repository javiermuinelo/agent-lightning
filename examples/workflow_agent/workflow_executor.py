# Copyright (c) Microsoft. All rights reserved.

"""
Workflow Executor for Agent-Workflow Contrastive RL.

This module provides functionality to execute generated LangGraph workflows
by injecting required dependencies (like LLM instances) and invoking the
compiled graphs with tasks.
"""

import os
from typing import Any, Dict

import dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Literal, TypedDict, Annotated

# Load environment variables
dotenv.load_dotenv()


def execute_workflow(
    workflow_code: str,
    task: str,
    model_name: str,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Execute a generated LangGraph workflow with a given task.
    
    This function:
    1. Creates an LLM instance (ChatHuggingFace with HuggingFaceEndpoint)
    2. Executes the workflow code in a namespace with the LLM and required imports
    3. Extracts the compiled graph from the executed code
    4. Invokes the graph with the task
    5. Returns the raw output state
    
    Args:
        workflow_code: Python code string defining a LangGraph workflow
                      (already extracted from <output> tags)
        task: The task description/input to pass to the workflow
        model_name: HuggingFace model identifier
        temperature: Temperature parameter for LLM generation
    
    Returns:
        Dictionary containing the final state after graph execution
        
    Raises:
        Any exceptions from code execution, compilation, or graph invocation
        are propagated to the caller.
    """
    # Initialize the LLM instance (same as in workflow_agent.py)
    llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
        model=model_name,
        task="text-generation",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        temperature=temperature,
        max_new_tokens=32768,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.03,
        provider="auto",  # Let HuggingFace choose the best provider
    ))
    
    # Prepare execution namespace with all necessary imports and the LLM
    execution_namespace = {
        # LLM instance
        "llm": llm,
        
        # Common imports needed by generated workflows
        "StateGraph": StateGraph,
        "START": START,
        "END": END,
        "TypedDict": TypedDict,
        "Annotated": Annotated,
        "Literal": Literal,
        "add_messages": add_messages,
        
        # Standard library imports that might be needed
        "os": os,
        "Dict": Dict,
        "Any": Any,
    }
    
    # Execute the workflow code
    # This will define the State class, node functions, and compile the graph
    exec(workflow_code, execution_namespace)
    
    # Extract the compiled graph from the namespace
    graph = execution_namespace.get("graph")
    
    if graph is None:
        raise ValueError(
            "Workflow code did not produce a 'graph' variable. "
            "Expected the code to compile and assign a graph to the 'graph' variable."
        )
    
    # Invoke the graph with the task
    # The workflow is expected to handle the task appropriately
    # Most workflows will have a 'task' field in their State
    # Type ignore needed because graph is dynamically extracted from exec()
    result: Dict[str, Any] = graph.invoke({"task": task})  # type: ignore
    
    return result

