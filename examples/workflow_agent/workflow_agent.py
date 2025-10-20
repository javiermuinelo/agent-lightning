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

import dotenv
import ray
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import agentlightning
from agentlightning.types import Rollout

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)

# Import prompts from separate module
from prompts import (
    AGENT_GENERATION_PROMPT,
    WORKFLOW_GENERATION_PROMPT,
    LLM_JUDGE_PROMPT,
    get_variation_strategy,
    format_agent_exemplars,
    format_workflow_exemplars,
)

# Import workflow executor
from workflow_executor import execute_workflow

# Import bucket manager
from buckets import get_or_create_bucket_manager # type:ignore


# ============================================================================
# Helper Classes
# ============================================================================

class SimpleMessage:
    """Simple message wrapper for HuggingFace responses."""
    
    def __init__(self, content: str) -> None:
        self.content = content


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(TypedDict):
    """State for the Agent-Workflow graph."""
    
    # Input
    task: str
    ground_truth: Optional[str]  # Ground truth answer for scoring
    
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
    
    # Final answers from workflows
    workflow_1_final_answer: Optional[str]
    workflow_2_final_answer: Optional[str]
    workflow_3_final_answer: Optional[str]
    
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
        use_huggingface: bool = False,
        bucket_manager: Optional[ray.actor.ActorHandle] = None, # type: ignore
    ):
        self.debug = debug
        self.bucket_manager = bucket_manager # type: ignore
        
        # Initialize LLM
        if use_huggingface:
            # Use HuggingFace Inference Providers via init_chat_model
            model_name = os.environ.get("MODEL", "Qwen/Qwen3-8B")
            
            temperature = 0.7
            
            self.model_name = model_name
            
            logger.info(f"Initializing HuggingFace model: {self.model_name}")
            
            # Use HuggingFaceEndpoint directly - this is the correct approach
            
            self.llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
                model=self.model_name,  # Use model parameter instead of repo_id
                task="text-generation",
                huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                temperature=temperature,
                max_new_tokens=4096,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.03,
                timeout=300,
                provider="nscale",  # Let HuggingFace choose the best provider
            ))
            
        elif verl_replacement is not None:
            # Use OpenAI-compatible endpoint (for VERL training)
            self.model_name: str = verl_replacement["model"] # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=max_retries,
                max_tokens=32768,
            )
        else:
            # Use standard OpenAI endpoint
            self.model_name: str = os.environ.get("MODEL", "gpt-4o-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                temperature=0.7,
                max_retries=1,
                max_tokens=32768,
            )

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt and handle errors."""
        if self.debug:
            if hasattr(prompt, 'messages'):
                for message in prompt.messages:
                    print(f"[PROMPT] {message.content[:200]}...")
            else:
                print(f"[PROMPT] {str(prompt)[:200]}...")
        
        try:
            # Check if we're using HuggingFace (which expects string input)
            if hasattr(self.llm, 'repo_id'):  # HuggingFaceEndpoint has repo_id attribute
                # Convert prompt to string for HuggingFace
                if hasattr(prompt, 'messages'):
                    prompt_text = "\n\n".join([msg.content for msg in prompt.messages if hasattr(msg, 'content')])
                else:
                    prompt_text = str(prompt)
                
                result_text = self.llm.invoke(prompt_text)
                
                # Use the global SimpleMessage class
                result = SimpleMessage(str(result_text))
            else:
                # Standard LangChain interface
                result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            if hasattr(self.llm, 'repo_id'):  # HuggingFace
                result_text = self.llm.invoke("Generate a simple workflow.")
                result = SimpleMessage(str(result_text))
            else:
                result = self.llm.invoke([{"role": "user", "content": "Generate a simple workflow."}])
        
        if self.debug:
            if hasattr(result, 'content'):
                print(f"[RESPONSE] {result.content[:200]}...") # type: ignore
            else:
                print(f"[RESPONSE] {str(result)[:200]}...")
        
        return result # type: ignore

    def extract_output_tag(self, content: str) -> str:
        """Extract content from <output> tags if present, otherwise return full content."""
        import re
        
        # Try to find content within <output> tags
        output_match = re.search(r'<output>\s*(.*?)\s*</output>', content, re.DOTALL | re.IGNORECASE)
        if output_match:
            return output_match.group(1)
        
        return content
    
    def parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response (looks in <output> tags first)."""
        import re
        
        # First, extract content from <output> tags if present
        output_content = self.extract_output_tag(content)
        
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', output_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to parse entire content as JSON
        try:
            return json.loads(output_content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
            return None
    
    def parse_python_code(self, content: str) -> Optional[str]:
        """Extract Python code from LLM response (looks in <output> tags first)."""
        import re
        
        # First, extract content from <output> tags if present
        output_content = self.extract_output_tag(content)
        
        # Try to find Python code block
        python_match = re.search(r'```python\s*(.*?)\s*```', output_content, re.DOTALL)
        if python_match:
            return python_match.group(1).strip()
        
        # If no code block found, return None
        logger.warning("Failed to extract Python code from response")
        return None
    
    # ========================================================================
    # Graph Nodes (Trainable: generate_agent, generate_workflow_*)
    # ========================================================================

    def generate_agent(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Generate meta-agent specification (TRAINABLE NODE).
        
        This node generates high-level search operators, constraints, and guidance
        that will be used to generate workflows.
        """
        logger.info(f"[Agent Node] Generating meta-agent for task: {state['task'][:50]}...")
        
        # Sample agent exemplars from bucket manager if available
        exemplars = None
        if self.bucket_manager is not None: # type: ignore
            try:
                exemplars = ray.get(self.bucket_manager.sample_agent_exemplars.remote()) # type: ignore
                if exemplars:
                    logger.info(f"[Agent Node] Sampled {len(exemplars)} agent exemplars from buckets")
            except Exception as e:
                logger.warning(f"[Agent Node] Failed to sample agent exemplars: {e}")
        
        # Format exemplars for prompt injection
        exemplars_section = format_agent_exemplars(exemplars)
        
        prompt = AGENT_GENERATION_PROMPT.invoke({ # type: ignore
            "task": state["task"],
            "exemplars_section": exemplars_section,
        })
        
        result = self.invoke_prompt(prompt)
        agent_output = result.content # type: ignore
        agent_output_parsed = self.parse_json_response(agent_output)    # type: ignore
        
        logger.info(f"[Agent Node] Generated agent output: {agent_output[:100]}...")
        
        return { # type: ignore
            "agent_output": agent_output, # type: ignore
            "agent_output_parsed": agent_output_parsed,
        }

    def generate_workflow_1(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate first workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=1, key="workflow_1")

    def generate_workflow_2(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate second workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=2, key="workflow_2")

    def generate_workflow_3(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate third workflow candidate (TRAINABLE NODE)."""
        return self._generate_workflow(state, workflow_id=3, key="workflow_3")

    def _generate_workflow(
        self, state: WorkflowState, workflow_id: int, key: str
    ) -> Dict[str, Any]:
        """
        Generate a workflow based on agent guidance (TRAINABLE NODE).
        
        This is called by generate_workflow_1/2/3 in parallel.
        Each workflow_id gets a different variation strategy to ensure diversity.
        """
        logger.info(f"[Workflow Node {workflow_id}] Generating workflow with variation strategy {workflow_id}...")
        
        # Sample workflow exemplars from bucket manager if available
        exemplars = None
        if self.bucket_manager is not None: # type: ignore
            try:
                exemplars = ray.get(self.bucket_manager.sample_workflow_exemplars.remote()) # type: ignore
                if exemplars:
                    logger.info(f"[Workflow Node {workflow_id}] Sampled {len(exemplars)} workflow exemplars from buckets")
            except Exception as e:
                logger.warning(f"[Workflow Node {workflow_id}] Failed to sample workflow exemplars: {e}")
        
        # Format exemplars for prompt injection
        exemplars_section = format_workflow_exemplars(exemplars)
        
        prompt = WORKFLOW_GENERATION_PROMPT.invoke({ # type: ignore
            "task": state["task"],
            "agent_output": state["agent_output"],
            "variation_strategy": get_variation_strategy(workflow_id),
            "exemplars_section": exemplars_section,
        })
        
        result = self.invoke_prompt(prompt)
        workflow_output = result.content # type: ignore
        
        logger.info(f"[Workflow Node {workflow_id}] Generated workflow: {workflow_output[:100]}...")
        
        return { # type: ignore
            key: workflow_output,
        }

    # ========================================================================
    # Graph Nodes (Non-trainable: execution and aggregation)
    # ========================================================================

    def execute_workflow_1(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute first workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=1, key="workflow_1")

    def execute_workflow_2(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute second workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=2, key="workflow_2")

    def execute_workflow_3(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute third workflow (NON-TRAINABLE)."""
        return self._execute_workflow(state, workflow_id=3, key="workflow_3")

    def _execute_workflow(
        self, state: WorkflowState, workflow_id: int, key: str
    ) -> Dict[str, Any]:
        """
        Execute a workflow and extract its final answer.
        
        Parses the Python code from the workflow output and executes it using
        the workflow executor. Returns the result and extracted final_answer.
        Scoring is deferred to aggregate_results.
        """
        logger.info(f"[Execute {workflow_id}] Executing workflow...")
        
        workflow_content = state[key] # type: ignore
        
        # Parse Python code from the workflow output
        workflow_code = self.parse_python_code(workflow_content) # type: ignore
        
        if workflow_code is None:
            logger.error(f"[Execute {workflow_id}] Failed to parse Python code from workflow")
            # Return error state
            return { # type: ignore
                f"{key}_result": None,
                f"{key}_final_answer": None,
            }
        
        # Execute the workflow using the executor
        try:
            result = execute_workflow(
                workflow_code=workflow_code,
                task=state["task"],
                model_name=self.model_name,
                temperature=0.7,
            )
            logger.info(f"[Execute {workflow_id}] Workflow executed successfully")
            logger.info(f"[Execute {workflow_id}] Result: {str(result)}...")
            
            # Extract final_answer directly from workflow result
            final_answer = result.get("final_answer")
            if final_answer:
                logger.info(f"[Execute {workflow_id}] Final answer: {final_answer}")
            else:
                logger.warning(f"[Execute {workflow_id}] No final_answer field found in result")
            
        except Exception as e:
            logger.error(f"[Execute {workflow_id}] Workflow execution failed: {str(e)}")
            # Return error state
            return { # type: ignore
                f"{key}_result": {"error": str(e), "workflow": workflow_content},
                f"{key}_final_answer": None,
            }
        
        result_key = f"workflow_{workflow_id}_result"
        final_answer_key = f"workflow_{workflow_id}_final_answer"
        
        return { # type: ignore
            result_key: {"executed": True, "result": result, "workflow": workflow_content, "code": workflow_code},
            final_answer_key: final_answer,
        }

    def aggregate_results(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Aggregate results from all workflows using LLM-as-a-judge (NON-TRAINABLE).
        
        Uses an LLM to evaluate each workflow's final answer against the ground truth,
        then selects the best workflow based on scores.
        """
        logger.info("[Aggregate] Evaluating workflow answers with LLM judge...")
        
        # Collect final answers and ground truth
        task = state["task"]
        ground_truth = state.get("ground_truth")
        workflow_1_answer = state.get("workflow_1_final_answer") or "No answer provided"
        workflow_2_answer = state.get("workflow_2_final_answer") or "No answer provided"
        workflow_3_answer = state.get("workflow_3_final_answer") or "No answer provided"
        
        # If no ground truth, return all zeros
        if not ground_truth:
            logger.warning("[Aggregate] No ground truth provided, assigning zero scores")
            return {
                "workflow_1_score": 0.0,
                "workflow_2_score": 0.0,
                "workflow_3_score": 0.0,
                "best_workflow": state.get("workflow_1"),
                "final_score": 0.0,
                "agent_reward": 0.0,
            }
        
        # Create judge prompt
        prompt = LLM_JUDGE_PROMPT.invoke({ # type: ignore
            "task": task,
            "ground_truth": ground_truth,
            "workflow_1_answer": workflow_1_answer,
            "workflow_2_answer": workflow_2_answer,
            "workflow_3_answer": workflow_3_answer,
        })
        
        # Invoke LLM judge
        try:
            result = self.invoke_prompt(prompt)
            judge_response = result.content # type: ignore
            logger.info(f"[Aggregate] Judge response: {judge_response[:200]}...")
            
            # Parse JSON response
            scores_dict = self.parse_json_response(judge_response) # type: ignore
            
            if scores_dict is None:
                logger.warning("[Aggregate] Failed to parse judge response, defaulting to zeros")
                scores_dict = {"workflow_1": 0, "workflow_2": 0, "workflow_3": 0}
            
            # Extract scores (ensure they are 0 or 1)
            workflow_1_score = float(scores_dict.get("workflow_1", 0))
            workflow_2_score = float(scores_dict.get("workflow_2", 0))
            workflow_3_score = float(scores_dict.get("workflow_3", 0))
            
        except Exception as e:
            logger.error(f"[Aggregate] Error invoking LLM judge: {e}")
            workflow_1_score = 0.0
            workflow_2_score = 0.0
            workflow_3_score = 0.0
        
        
        # Compute agent-level reward as average of all workflows
        agent_reward = (workflow_1_score + workflow_2_score + workflow_3_score) / 3.0
        
        logger.info(f"[Aggregate] Scores - W1: {workflow_1_score:.1f}, W2: {workflow_2_score:.1f}, W3: {workflow_3_score:.1f}")
        logger.info(f"[Aggregate] Agent reward (avg): {agent_reward:.3f}")
        
        # Add agent and workflows to bucket manager if available
        if self.bucket_manager is not None: # type: ignore
            try:
                # Add agent with its reward
                ray.get(self.bucket_manager.add_agent.remote( # type: ignore
                    state["agent_output_parsed"],
                    agent_reward
                ))
                logger.info(f"[Aggregate] Added agent to bucket (score={agent_reward:.3f})")
                
                # Add all three workflows with their scores
                if state.get("workflow_1_result") and state.get("workflow_1_result").get("code"): # type: ignore
                    ray.get(self.bucket_manager.add_workflow.remote( # type: ignore
                        state["workflow_1_result"].get("code"), # type: ignore
                        workflow_1_score
                    ))
                if state.get("workflow_2_result") and state.get("workflow_2_result").get("code"): # type: ignore
                    ray.get(self.bucket_manager.add_workflow.remote( # type: ignore
                        state["workflow_2_result"].get("code"), # type: ignore
                        workflow_2_score
                    ))
                if state.get("workflow_3_result") and state.get("workflow_3_result").get("code"): # type: ignore
                    ray.get(self.bucket_manager.add_workflow.remote( # type: ignore
                        state["workflow_3_result"].get("code"), # type: ignore
                        workflow_3_score
                    ))
                logger.info(f"[Aggregate] Added 3 workflows to bucket")
                
                # Log bucket stats
                stats = ray.get(self.bucket_manager.get_stats.remote()) # type: ignore
                logger.info(f"[Aggregate] Bucket stats: {stats}")
            except Exception as e:
                logger.warning(f"[Aggregate] Failed to add to buckets: {e}")
        
        return {
            "workflow_1_score": workflow_1_score,
            "workflow_2_score": workflow_2_score,
            "workflow_3_score": workflow_3_score,
            "agent_reward": agent_reward,
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
        
        # # Parallel workflow execution
        builder.add_edge("generate_workflow_1", "execute_workflow_1")
        builder.add_edge("generate_workflow_2", "execute_workflow_2")
        builder.add_edge("generate_workflow_3", "execute_workflow_3")
        
        # # Aggregate results
        builder.add_edge("execute_workflow_1", "aggregate_results")
        builder.add_edge("execute_workflow_2", "aggregate_results")
        builder.add_edge("execute_workflow_3", "aggregate_results")
        
        builder.add_edge("aggregate_results", END)
        
        return builder.compile() # type: ignore


# ============================================================================
# Agent-Lightning Integration
# ============================================================================

class LitWorkflowAgent(agentlightning.LitAgent): # type: ignore
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
        enable_buckets: bool = True,
        bucket_min_agent_score: float = 0.0,
        bucket_min_workflow_score: float = 0.0,
        bucket_temperature_agent: float = 1.0,
        bucket_temperature_workflow: float = 1.0,
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
            enable_buckets: Whether to enable bucket manager for contrastive learning
            bucket_min_agent_score: Minimum score threshold for adding agents to bucket
            bucket_min_workflow_score: Minimum score threshold for adding workflows to bucket
            bucket_temperature_agent: Temperature for agent bucket sampling (tau_A)
            bucket_temperature_workflow: Temperature for workflow bucket sampling (tau_W)
        """
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.debug = debug
        self.enable_buckets = enable_buckets
        self.bucket_min_agent_score = bucket_min_agent_score
        self.bucket_min_workflow_score = bucket_min_workflow_score
        self.bucket_temperature_agent = bucket_temperature_agent
        self.bucket_temperature_workflow = bucket_temperature_workflow
        self._bucket_manager = None

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
        
        # Initialize bucket manager if enabled and not already initialized
        bucket_manager = None
        if self.enable_buckets:
            if self._bucket_manager is None: # type: ignore
                try:
                    if ray.is_initialized(): # type: ignore
                        self._bucket_manager = get_or_create_bucket_manager( # type: ignore
                            min_agent_score=self.bucket_min_agent_score,
                            min_workflow_score=self.bucket_min_workflow_score,
                            temperature_agent=self.bucket_temperature_agent,
                            temperature_workflow=self.bucket_temperature_workflow,
                        )
                        logger.info(f"[Rollout {rollout_id}] Initialized bucket manager")
                    else:
                        logger.warning(f"[Rollout {rollout_id}] Ray not initialized, buckets disabled")
                        self.enable_buckets = False
                except Exception as e:
                    logger.warning(f"[Rollout {rollout_id}] Failed to initialize bucket manager: {e}")
                    self.enable_buckets = False
            bucket_manager = self._bucket_manager # type: ignore
        
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
            bucket_manager=bucket_manager,
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

if __name__ == "__main__":
    dotenv.load_dotenv()
    agent, trainer = agentlightning.lightning_cli(
        LitWorkflowAgent, agentlightning.Trainer
    )
    trainer.fit(agent, os.environ["VERL_API_BASE"])

