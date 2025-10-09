# Copyright (c) Microsoft. All rights reserved.

"""
Agent-Workflow Contrastive RL implementation.

This module implements a hierarchical policy where:
1. An Agent node generates high-level search operators, specifications, and guides
2. Multiple Workflow nodes generate executable graphs in parallel based on the Agent output
3. Both Agent and Workflow generations are trainable via GRPO
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

import agentlightning
from agentlightning.types import Rollout

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)


# ============================================================================
# Prompts
# ============================================================================

AGENT_GENERATION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a meta-agent that generates high-level search strategies and specifications for solving complex tasks.

Given a task, you should generate:
1. **Search Operators**: What types of decompositions or approaches to try
2. **Constraints**: Any limitations or requirements for the solution
3. **Guidance**: High-level hints for workflow construction
4. **Examples**: Brief examples of similar problems and approaches

Your output will guide the generation of executable workflows.

## Output Format ##

Respond in JSON format:

```json
{{
  "search_operators": ["operator1", "operator2", ...],
  "constraints": ["constraint1", "constraint2", ...],
  "guidance": "High-level approach description",
  "examples": ["example1", "example2", ...]
}}
```
""".strip(),
        ),
        ("user", "Task: {task}\n\nGenerate the meta-agent specification:"),
    ]
)


WORKFLOW_GENERATION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a workflow generator that creates executable graphs for solving tasks.

Given:
- A task description
- Meta-agent guidance (search operators, constraints, etc.)

Generate a workflow as a Python-like pseudo-code or Mermaid graph that decomposes the task into steps.

## Output Format ##

Respond with a structured workflow:

```python
# Workflow Steps
def solve_task(input):
    # Step 1: [description]
    step1_output = ...
    
    # Step 2: [description]
    step2_output = ...
    
    # Step 3: [description]
    final_output = ...
    
    return final_output
```

Be specific about what each step does and how they connect.
""".strip(),
        ),
        (
            "user",
            """Task: {task}

Meta-Agent Guidance:
{agent_output}

Generate workflow (attempt {workflow_id}):""",
        ),
    ]
)


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(TypedDict):
    """State for the Agent-Workflow graph."""
    
    # Input
    task: str
    
    # Agent generation
    agent_output: str
    agent_output_parsed: Optional[Dict[str, Any]]
    
    # Workflow generations (parallel)
    workflow_1: Optional[str]
    workflow_2: Optional[str]
    workflow_3: Optional[str]
    
    # Execution results (placeholder for now)
    workflow_1_result: Optional[Any]
    workflow_2_result: Optional[Any]
    workflow_3_result: Optional[Any]
    
    # Scores (placeholder for now)
    workflow_1_score: Optional[float]
    workflow_2_score: Optional[float]
    workflow_3_score: Optional[float]
    
    # Final result
    best_workflow: Optional[str]
    final_score: Optional[float]
    agent_reward: Optional[float]  # NEW: Agent-level reward


# ============================================================================
# Agent-Workflow Graph
# ============================================================================

class AgentWorkflowGraph:
    """
    Hierarchical Agent-Workflow graph implementing the contrastive RL architecture.
    
    Graph structure:
    START -> generate_agent -> [generate_workflow_1, generate_workflow_2, generate_workflow_3]
          -> [execute_workflow_1, execute_workflow_2, execute_workflow_3]
          -> aggregate_results -> END
    
    Trainable nodes: generate_agent, generate_workflow_*
    """

    def __init__(
        self,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        debug: bool = False,
        max_retries: int = 0,
    ):
        self.debug = debug
        
        # Initialize LLM
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"] # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=max_retries,
                max_tokens=2048,
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "gpt-4o-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.7,
                max_retries=1,
                max_tokens=2048,
            )

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt and handle errors."""
        if self.debug:
            for message in prompt.messages:
                print(f"[PROMPT] {message.content[:200]}...")
        
        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            result = self.llm.invoke([{"role": "user", "content": "Generate a simple workflow."}])
        
        if self.debug:
            print(f"[RESPONSE] {result.content[:200]}...") # type: ignore
        
        return result # type: ignore

    def parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        import re
        
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to parse entire content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
            return None

    # ========================================================================
    # Graph Nodes (Trainable: generate_agent, generate_workflow_*)
    # ========================================================================

    def generate_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Generate meta-agent specification (TRAINABLE NODE).
        
        This node generates high-level search operators, constraints, and guidance
        that will be used to generate workflows.
        """
        logger.info(f"[Agent Node] Generating meta-agent for task: {state['task'][:50]}...")
        
        prompt = AGENT_GENERATION_PROMPT.invoke({ # type: ignore
            "task": state["task"]
        })
        
        result = self.invoke_prompt(prompt)
        agent_output = result.content # type: ignore
        agent_output_parsed = self.parse_json_response(agent_output)    # type: ignore
        
        logger.info(f"[Agent Node] Generated agent output: {agent_output[:100]}...")
        
        return { # type: ignore
            **state,
            "agent_output": agent_output, # type: ignore
            "agent_output_parsed": agent_output_parsed,
        }

    def generate_workflow_1(self, state: WorkflowState) -> WorkflowState:
        """Generate first workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=1, key="workflow_1")

    def generate_workflow_2(self, state: WorkflowState) -> WorkflowState:
        """Generate second workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=2, key="workflow_2")

    def generate_workflow_3(self, state: WorkflowState) -> WorkflowState:
        """Generate third workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=3, key="workflow_3")

    def _generate_workflow(
        self, state: WorkflowState, workflow_id: int, key: str
    ) -> WorkflowState:
        """
        Generate a workflow based on agent guidance (TRAINABLE NODE).
        
        This is called by generate_workflow_1/2/3 in parallel.
        """
        logger.info(f"[Workflow Node {workflow_id}] Generating workflow...")
        
        prompt = WORKFLOW_GENERATION_PROMPT.invoke({ # type: ignore
            "task": state["task"],
            "agent_output": state["agent_output"],
            "workflow_id": workflow_id,
        })
        
        result = self.invoke_prompt(prompt)
        workflow_output = result.content # type: ignore
        
        logger.info(f"[Workflow Node {workflow_id}] Generated workflow: {workflow_output[:100]}...")
        
        return { # type: ignore
            **state,
            key: workflow_output,
        }

    # ========================================================================
    # Graph Nodes (Non-trainable: execution and aggregation)
    # ========================================================================

    def execute_workflow_1(self, state: WorkflowState) -> WorkflowState:
        """Execute first workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=1, key="workflow_1")

    def execute_workflow_2(self, state: WorkflowState) -> WorkflowState:
        """Execute second workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=2, key="workflow_2")

    def execute_workflow_3(self, state: WorkflowState) -> WorkflowState:
        """Execute third workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=3, key="workflow_3")

    def _execute_workflow(
        self, state: WorkflowState, workflow_id: int, key: str
    ) -> WorkflowState:
        """
        Execute a workflow and compute its score (PLACEHOLDER).
        
        TODO: Implement actual workflow execution (Python/Mermaid interpreter)
        TODO: Implement LLM-as-Judge scoring
        """
        logger.info(f"[Execute {workflow_id}] Executing workflow (PLACEHOLDER)...")
        
        workflow_content = state[key] # type: ignore
        
        # PLACEHOLDER: Random score for now
        import random
        score = random.uniform(0.3, 1.0)
        
        result_key = f"workflow_{workflow_id}_result"
        score_key = f"workflow_{workflow_id}_score"
        
        logger.info(f"[Execute {workflow_id}] Score: {score:.3f}")
        
        return { # type: ignore
            **state,
            result_key: {"executed": True, "workflow": workflow_content},
            score_key: score,
        }

    def aggregate_results(self, state: WorkflowState) -> WorkflowState:
        """
        Aggregate results from all workflows and select the best (NON-TRAINABLE).
        """
        logger.info("[Aggregate] Selecting best workflow...")
        
        scores = [
            (1, state.get("workflow_1_score", 0.0), state.get("workflow_1")),
            (2, state.get("workflow_2_score", 0.0), state.get("workflow_2")),
            (3, state.get("workflow_3_score", 0.0), state.get("workflow_3")),
        ]
        
        # Select best workflow
        best_id, best_score, best_workflow = max(scores, key=lambda x: x[1]) # type: ignore
        
        # Compute agent-level reward as average of all workflows
        workflow_scores = [
            state.get("workflow_1_score") or 0.0,
            state.get("workflow_2_score") or 0.0,
            state.get("workflow_3_score") or 0.0,
        ]
        agent_reward = sum(workflow_scores) / 3.0
        
        logger.info(f"[Aggregate] Best workflow: {best_id} with score {best_score:.3f}")
        logger.info(f"[Aggregate] Agent reward (avg): {agent_reward:.3f}")
        
        return {
            **state,
            "best_workflow": best_workflow,
            "final_score": best_score,
            "agent_reward": agent_reward,  # NEW: Store agent reward explicitly
        }

    # ========================================================================
    # Graph Construction
    # ========================================================================

    def build_graph(self) -> CompiledStateGraph: # type: ignore
        """
        Build the Agent-Workflow graph with parallel workflow generation.
        
        Graph structure:
        START -> generate_agent
              -> [generate_workflow_1, generate_workflow_2, generate_workflow_3] (parallel)
              -> [execute_workflow_1, execute_workflow_2, execute_workflow_3] (parallel)
              -> aggregate_results
              -> END
        """
        builder = StateGraph(WorkflowState)
        
        # Add nodes
        builder.add_node("generate_agent", self.generate_agent) # type: ignore
        builder.add_node("generate_workflow_1", self.generate_workflow_1) # type: ignore
        builder.add_node("generate_workflow_2", self.generate_workflow_2) # type: ignore
        builder.add_node("generate_workflow_3", self.generate_workflow_3) # type: ignore
        builder.add_node("execute_workflow_1", self.execute_workflow_1) # type: ignore
        builder.add_node("execute_workflow_2", self.execute_workflow_2) # type: ignore
        builder.add_node("execute_workflow_3", self.execute_workflow_3) # type: ignore
        builder.add_node("aggregate_results", self.aggregate_results) # type: ignore
        
        # Connect edges
        builder.add_edge(START, "generate_agent")
        
        # Parallel workflow generation
        builder.add_edge("generate_agent", "generate_workflow_1") # type: ignore
        builder.add_edge("generate_agent", "generate_workflow_2")
        builder.add_edge("generate_agent", "generate_workflow_3")
        
        # Parallel workflow execution
        builder.add_edge("generate_workflow_1", "execute_workflow_1")
        builder.add_edge("generate_workflow_2", "execute_workflow_2")
        builder.add_edge("generate_workflow_3", "execute_workflow_3")
        
        # Aggregate results
        builder.add_edge("execute_workflow_1", "aggregate_results")
        builder.add_edge("execute_workflow_2", "aggregate_results")
        builder.add_edge("execute_workflow_3", "aggregate_results")
        
        builder.add_edge("aggregate_results", END)
        
        return builder.compile() # type: ignore


# ============================================================================
# Agent-Lightning Integration
# ============================================================================

class LitWorkflowAgent(agentlightning.LitAgent[Any]):
    """
    Agent-Lightning wrapper for the Agent-Workflow graph.
    
    This class integrates the hierarchical policy with agent-lightning's
    training infrastructure, enabling GRPO optimization of both the Agent
    and Workflow generation nodes.
    """

    def __init__(
        self,
        trained_agents: Optional[str] = r"generate_(agent|workflow_\d+)",
        val_temperature: Optional[float] = 0.0,
        debug: bool = False,
    ) -> None:
        """
        Initialize the LitWorkflowAgent.
        
        Args:
            trained_agents: Regex pattern matching trainable nodes.
                           Default: r"generate_(agent|workflow_x)"
                           This matches: generate_agent, generate_workflow_1, 
                                       generate_workflow_2, generate_workflow_3
            val_temperature: Temperature for validation rollouts
            debug: Enable debug logging
        """
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.debug = debug

    def _execute_rollout(
        self,
        sample: dict[str, Any],
        *,
        resources: agentlightning.NamedResources,
        rollout_id: str,
        is_training: bool,
    ) -> Rollout | float | None:
        """
        Execute a single rollout of the Agent-Workflow graph.
        
        Returns:
            Rollout object with metadata containing node-specific rewards
        """
        task = sample["task"]
        start_time = time.time()
        
        llm: agentlightning.LLM = cast(agentlightning.LLM, resources["main_llm"])
        
        logger.info(f"[Rollout {rollout_id}] Task: {task[:100]}...")
        
        # Build graph with appropriate temperature
        graph = AgentWorkflowGraph( # type: ignore
            endpoint=llm.endpoint,
            verl_replacement=(
                {"model": llm.model, **llm.sampling_parameters}
                if is_training
                else {
                    "model": llm.model,
                    "temperature": (
                        self.val_temperature
                        if self.val_temperature is not None
                        else llm.sampling_parameters.get("temperature", 0.0)
                    ),
                }
            ),
            debug=self.debug,
        ).build_graph()
        
        try:
            # Execute graph with tracer callback
            result = graph.invoke( # type: ignore
                {"task": task},
                {
                    "callbacks": [self.tracer.get_langchain_callback_handler()], # type: ignore
                    "recursion_limit": 100,
                },
            )
        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during graph execution: {e}")
            return None
        
        end_time = time.time()
        
        # Extract scores
        agent_reward = result.get("agent_reward", 0.0)
        final_score = result.get("final_score", 0.0)
        best_workflow = result.get("best_workflow", "N/A")
        
        logger.info(f"[Rollout {rollout_id}] Agent Reward: {agent_reward:.3f}")
        logger.info(f"[Rollout {rollout_id}] Final Score: {final_score:.3f}")
        logger.info(f"[Rollout {rollout_id}] Best Workflow: {best_workflow[:100]}...")
        logger.info(f"[Rollout {rollout_id}] Time: {end_time - start_time:.2f}s")
        
        # Return Rollout with structured rewards in metadata
        # This allows the daemon to assign different rewards to different nodes
        return Rollout(
            rollout_id=rollout_id,
            final_reward=agent_reward,  # Use agent_reward as primary reward
            metadata={
                "node_rewards": {
                    "agent_reward": agent_reward,
                    "workflow_1_score": result.get("workflow_1_score", 0.0),
                    "workflow_2_score": result.get("workflow_2_score", 0.0),
                    "workflow_3_score": result.get("workflow_3_score", 0.0),
                }
            },
        )

    def training_rollout( # type: ignore
        self, task: Any, rollout_id: str, resources: agentlightning.NamedResources
    ) -> Any:
        """Execute training rollout with sampling temperature."""
        return self._execute_rollout(
            task, resources=resources, rollout_id=rollout_id, is_training=True
        )

    def validation_rollout( # type: ignore
        self, task: Any, rollout_id: str, resources: agentlightning.NamedResources
    ) -> Any:
        """Execute validation rollout with lower temperature."""
        return self._execute_rollout(
            task, resources=resources, rollout_id=rollout_id, is_training=False
        )


# ============================================================================
# Dev Data Loader (for testing)
# ============================================================================

# def dev_data():
#     """Load development data for testing."""
#     # Sample tasks for testing
#     sample_tasks = [
#         {
#             "task": "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes algorithm."
#         },
#         {
#             "task": "Create a recursive solution to solve the Tower of Hanoi problem with n disks."
#         },
#         {
#             "task": "Implement a binary search tree with insert, delete, and search operations."
#         },
#         {
#             "task": "Write a dynamic programming solution for the knapsack problem."
#         },
#         {
#             "task": "Create a graph traversal algorithm to find the shortest path between two nodes."
#         },
#     ]
    
#     if "OPENAI_API_BASE" not in os.environ:
#         logger.warning(
#             "Environment variable OPENAI_API_BASE is not set. "
#             "Using default value 'https://api.openai.com/v1'."
#         )
#         openai_api_base = "https://api.openai.com/v1"
#     else:
#         openai_api_base = os.environ["OPENAI_API_BASE"]
    
#     resource = {
#         "main_llm": agentlightning.LLM(
#             model="gpt-4o-mini",
#             endpoint=openai_api_base,
#             sampling_parameters={
#                 "temperature": 0.7,
#             },
#         )
#     }
    
#     return agentlightning.DevTaskLoader(sample_tasks, resource)


# # ============================================================================
# # Main Entry Point
# # ============================================================================

# if __name__ == "__main__":
#     dotenv.load_dotenv()
#     agent, trainer = agentlightning.lightning_cli(
#         LitWorkflowAgent, agentlightning.Trainer
#     )
#     trainer.fit(agent, os.environ["VERL_API_BASE"], dev_data=dev_data())

