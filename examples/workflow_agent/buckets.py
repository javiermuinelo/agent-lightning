# Copyright (c) Microsoft. All rights reserved.

"""
Bucket Manager for Agent-Workflow Contrastive RL.

This module implements a Ray Actor-based bucket system for storing and sampling
agent and workflow exemplars with scores. Buckets are partitioned into Fast/Medium/Slow
based on score quantiles and sampled contrastively for prompt injection.

Implements Section 4.3-4.4 of the research proposal.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import ray

import agentlightning

logger = agentlightning.configure_logger(name=__name__)


@ray.remote
class BucketManager:
    """
    Ray Actor for managing agent and workflow buckets across distributed training.
    
    Stores (output, score) tuples for agents and workflows, partitions them into
    Fast/Medium/Slow buckets based on score quantiles, and provides contrastive
    sampling for prompt exemplar injection.
    
    This is a singleton Ray Actor shared across all workers in the cluster.
    """
    
    def __init__(
        self,
        min_agent_score: float = 0.0,
        min_workflow_score: float = 0.0,
        temperature_agent: float = 1.0,
        temperature_workflow: float = 1.0,
    ):
        """
        Initialize the BucketManager.
        
        Args:
            min_agent_score: Minimum score threshold for adding agents to bucket
            min_workflow_score: Minimum score threshold for adding workflows to bucket
            temperature_agent: Temperature for agent bucket sampling (tau_A)
            temperature_workflow: Temperature for workflow bucket sampling (tau_W)
        """
        self.agent_buffer: List[Tuple[str, float]] = []
        self.workflow_buffer: List[Tuple[str, float]] = []
        
        self.min_agent_score = min_agent_score
        self.min_workflow_score = min_workflow_score
        self.tau_A = temperature_agent
        self.tau_W = temperature_workflow
        
        logger.info(
            f"BucketManager initialized: "
            f"min_agent_score={min_agent_score}, min_workflow_score={min_workflow_score}, "
            f"tau_A={temperature_agent}, tau_W={temperature_workflow}"
        )
    
    def add_agent(self, agent_output: str, score: float) -> None:
        """
        Add an agent to the agent bucket if it meets the minimum score threshold.
        
        Args:
            agent_output: The agent's output (JSON string or raw output)
            score: The agent-level reward (mean of workflow scores)
        """
        if score >= self.min_agent_score:
            self.agent_buffer.append((agent_output, score))
            logger.info(
                f"Added agent to bucket (score={score:.3f}, total_agents={len(self.agent_buffer)})"
            )
        else:
            logger.debug(
                f"Agent score {score:.3f} below threshold {self.min_agent_score}, not added"
            )
    
    def add_workflow(self, workflow_output: str, score: float) -> None:
        """
        Add a workflow to the workflow bucket if it meets the minimum score threshold.
        
        Args:
            workflow_output: The workflow's output (Python code string)
            score: The workflow's score (0 or 1 from LLM judge)
        """
        if score >= self.min_workflow_score:
            self.workflow_buffer.append((workflow_output, score))
            logger.info(
                f"Added workflow to bucket (score={score:.3f}, total_workflows={len(self.workflow_buffer)})"
            )
        else:
            logger.debug(
                f"Workflow score {score:.3f} below threshold {self.min_workflow_score}, not added"
            )
    
    def _partition_buckets(
        self, buffer: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Partition a buffer into Fast/Medium/Slow buckets based on score quantiles.
        
        Following Section 4.3 of the research proposal:
        - Fast: score >= T_F (66.67th percentile)
        - Medium: T_S < score < T_F
        - Slow: score <= T_S (33.33th percentile)
        
        Args:
            buffer: List of (output, score) tuples
            
        Returns:
            Tuple of (Fast, Medium, Slow) bucket lists
        """
        if len(buffer) == 0:
            return [], [], []
        
        if len(buffer) == 1:
            # Single item goes to Fast bucket
            return [buffer[0]], [], []
        
        if len(buffer) == 2:
            # Two items: higher goes to Fast, lower to Slow
            sorted_buffer = sorted(buffer, key=lambda x: x[1], reverse=True)
            return [sorted_buffer[0]], [], [sorted_buffer[1]]
        
        # Extract scores for percentile calculation
        scores = [score for _, score in buffer]
        
        # Calculate thresholds at 66.67% and 33.33% quantiles
        T_F = np.percentile(scores, 66.67)
        T_S = np.percentile(scores, 33.33)
        
        # Partition into buckets
        Fast = [(item, score) for item, score in buffer if score >= T_F]
        Medium = [(item, score) for item, score in buffer if T_S < score < T_F]
        Slow = [(item, score) for item, score in buffer if score <= T_S]
        
        return Fast, Medium, Slow
    
    def _softmax(self, logits: List[float]) -> np.ndarray:  # type: ignore
        """Compute softmax probabilities from logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()  # type: ignore
    
    def _sample_contrastive_exemplars(
        self, buffer: List[Tuple[str, float]], tau: float
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Sample 2 contrastive exemplars from the buffer using the algorithm from Section 4.3.
        
        Algorithm:
        1. Partition buffer into Fast/Medium/Slow buckets
        2. Compute mean score for each bucket
        3. Weight buckets using softmax of (mean - global_mean) / tau
        4. Sample 2 distinct buckets according to weights
        5. Sample 1 exemplar uniformly from each bucket
        
        Args:
            buffer: List of (output, score) tuples
            tau: Temperature for softmax weighting
            
        Returns:
            List of 2 (output, score) tuples, or None if sampling is not possible
        """
        if len(buffer) < 2:
            logger.warning(
                f"Buffer has {len(buffer)} items, need at least 2 for contrastive sampling"
            )
            return None
        
        # Partition into buckets
        Fast, Medium, Slow = self._partition_buckets(buffer)
        buckets = [Fast, Medium, Slow]
        bucket_names = ["Fast", "Medium", "Slow"]
        
        # Filter out empty buckets
        non_empty_buckets = [(b, name) for b, name in zip(buckets, bucket_names) if len(b) > 0]
        
        if len(non_empty_buckets) < 2:
            logger.warning(
                f"Only {len(non_empty_buckets)} non-empty buckets, need at least 2 for contrastive sampling"
            )
            # Fallback: sample 2 random items from the buffer
            return random.sample(buffer, 2)
        
        # Compute bucket means
        bucket_means: List[float] = []
        for bucket, _ in non_empty_buckets:
            mean_score = np.mean([score for _, score in bucket])
            bucket_means.append(float(mean_score))
        # Compute global mean
        all_scores = [score for _, score in buffer]
        global_mean = np.mean(all_scores)
        
        # Compute softmax weights centered by global mean
        # P^A_t(B^A_κ) = exp((q̃^A_κ - μ^A_t) / tau) / Σ exp(...)
        logits = [(mean - global_mean) / tau for mean in bucket_means]
        weights = self._softmax(logits) # type: ignore
        
        logger.debug(
            f"Bucket sampling - sizes: {[len(b) for b, _ in non_empty_buckets]}, "
            f"means: {[f'{m:.3f}' for m in bucket_means]}, "
            f"weights: {[f'{w:.3f}' for w in weights]}"  # type: ignore
        )
        
        # Sample 2 distinct buckets
        selected_indices = np.random.choice(  # type: ignore
            len(non_empty_buckets), size=min(2, len(non_empty_buckets)), replace=False, p=weights # type: ignore
        )
        
        # Sample 1 exemplar from each selected bucket
        exemplars: List[Tuple[str, float]] = []
        for idx in selected_indices:  # type: ignore
            bucket, bucket_name = non_empty_buckets[idx] # type: ignore
            exemplar = random.choice(bucket) # type: ignore
            exemplars.append(exemplar) # type: ignore
            logger.debug(f"Sampled from {bucket_name} bucket: score={exemplar[1]:.3f}")
        
        return exemplars
    
    def sample_agent_exemplars(self) -> Optional[List[Tuple[str, float]]]:
        """
        Sample 2 contrastive agent exemplars for prompt injection.
        
        Returns:
            List of 2 (agent_output, score) tuples, or None if not enough agents
        """
        logger.info(f"Sampling agent exemplars from {len(self.agent_buffer)} agents")
        return self._sample_contrastive_exemplars(self.agent_buffer, self.tau_A)
    
    def sample_workflow_exemplars(self) -> Optional[List[Tuple[str, float]]]:
        """
        Sample 2 contrastive workflow exemplars for prompt injection.
        
        Returns:
            List of 2 (workflow_output, score) tuples, or None if not enough workflows
        """
        logger.info(f"Sampling workflow exemplars from {len(self.workflow_buffer)} workflows")
        return self._sample_contrastive_exemplars(self.workflow_buffer, self.tau_W)
    

def get_or_create_bucket_manager( # type: ignore
    min_agent_score: float = 0.0,
    min_workflow_score: float = 0.0,
    temperature_agent: float = 1.0,
    temperature_workflow: float = 1.0,
) -> ray.actor.ActorHandle:  # type: ignore
    """
    Get the global BucketManager Ray Actor, creating it if it doesn't exist.
    
    This function ensures a singleton BucketManager exists in the Ray cluster.
    
    Args:
        min_agent_score: Minimum score threshold for adding agents
        min_workflow_score: Minimum score threshold for adding workflows
        temperature_agent: Temperature for agent bucket sampling
        temperature_workflow: Temperature for workflow bucket sampling
        
    Returns:
        Ray Actor handle for the BucketManager
    """
    actor_name = "bucket_manager"
    
    try:
        # Try to get existing actor
        bucket_manager = ray.get_actor(actor_name)  # type: ignore
        logger.info(f"Retrieved existing BucketManager actor: {actor_name}")
        return bucket_manager  # type: ignore
    except ValueError:
        # Actor doesn't exist, create it
        logger.info(f"Creating new BucketManager actor: {actor_name}")
        bucket_manager = BucketManager.options(name=actor_name, lifetime="detached").remote(  # type: ignore
            min_agent_score=min_agent_score,
            min_workflow_score=min_workflow_score,
            temperature_agent=temperature_agent,
            temperature_workflow=temperature_workflow,
        )
        return bucket_manager  # type: ignore

