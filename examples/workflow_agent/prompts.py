# Copyright (c) Microsoft. All rights reserved.

"""
Prompts for Agent-Workflow Contrastive RL.

This module contains the prompt templates for:
1. Agent Generation: High-level task analysis, decomposition, and planning
2. Workflow Generation: Executable graph generation based on agent guidance
"""

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Variation Strategies for Workflow Diversity
# ============================================================================

WORKFLOW_VARIATION_STRATEGIES = {
    1: """
**STRATEGY 1: ROBUST & DEFENSIVE**
- Prioritize error handling and validation at each step
- Add explicit validation nodes between subtasks
- Use defensive state checks (e.g., check if fields exist before accessing)
- Prefer explicit conditional routing over assuming happy path
- Include fallback mechanisms when the agent identified edge cases
- Trade-off: More nodes/edges, potentially slower but more reliable
""".strip(),
    
    2: """
**STRATEGY 2: EFFICIENT & STREAMLINED**
- Minimize the number of nodes by combining simple operations
- Prefer direct linear flow with minimal conditionals
- Merge subtasks that have trivial implementations into single nodes
- Use simpler state structure with fewer intermediate fields
- Aggressively leverage parallel opportunities identified by the agent
- Trade-off: Faster execution, but less granular error tracking
""".strip(),
    
    3: """
**STRATEGY 3: MODULAR & DETAILED**
- Break complex subtasks into even smaller, atomic operations
- Create separate nodes for preparation, execution, and validation of each subtask
- Use rich state with many intermediate results for debugging
- Add monitoring/logging points as separate nodes where appropriate
- Prefer explicit over implicit - make all data flows visible in the graph
- Trade-off: More detailed but potentially more complex graph structure
""".strip(),
}


def get_variation_strategy(workflow_id: int) -> str:
    """
    Get the variation strategy prompt for a specific workflow ID.
    
    This ensures that each of the 3 workflows explores a different
    implementation approach, enabling contrastive learning.
    """
    return WORKFLOW_VARIATION_STRATEGIES.get(
        workflow_id,
        WORKFLOW_VARIATION_STRATEGIES[1]  # Default to strategy 1
    )


def format_agent_exemplars(exemplars: list[tuple[str, float]] | None) -> str:
    """
    Format agent exemplars for prompt injection.
    
    Args:
        exemplars: List of (agent_output, score) tuples, or None
        
    Returns:
        Formatted string to inject into prompt, or empty string if no exemplars
    """
    if exemplars is None or len(exemplars) == 0:
        return ""
    
    lines = ["Here are some example agent analyses from previous tasks:\n"]
    
    for i, (agent_output, score) in enumerate(exemplars, 1):
        lines.append(f"Example {i} (Score: {score:.3f}):")
        lines.append(f"{agent_output}\n")
    
    return "\n".join(lines)


def format_workflow_exemplars(exemplars: list[tuple[str, float]] | None) -> str:
    """
    Format workflow exemplars for prompt injection.
    
    Args:
        exemplars: List of (workflow_output, score) tuples, or None
        
    Returns:
        Formatted string to inject into prompt, or empty string if no exemplars
    """
    if exemplars is None or len(exemplars) == 0:
        return ""
    
    lines = ["Here are some example workflows from previous tasks:\n"]
    
    for i, (workflow_output, score) in enumerate(exemplars, 1):
        lines.append(f"Example {i} (Score: {score:.3f}):")
        lines.append(f"{workflow_output}\n")
    
    return "\n".join(lines)


# ============================================================================
# Agent Generation Prompt
# ============================================================================

AGENT_GENERATION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a strategic meta-agent responsible for deep task analysis and decomposition.

Your role is to analyze complex tasks and produce a comprehensive planning specification that will guide downstream workflow generation. You do NOT generate executable code or workflows yourself—instead, you provide the strategic foundation that makes workflow generation effective.

## Your Responsibilities ##

1. **Task Analysis**: Understand the core objective, implicit requirements, and success criteria
2. **Strategy Selection**: Determine high-level approaches (e.g., divide-and-conquer, iterative refinement, parallel processing)
3. **Constraint Identification**: Surface limitations, edge cases, and requirements
4. **Context Provision**: Provide relevant domain knowledge or analogies

## Output Format ##

First, think through your analysis in <plan> tags, then provide your final JSON output in <output> tags.

The JSON should have this structure:

```json
{{
  "objective": "Clear statement of the primary goal",
  "domain": "The domain this task belongs to (e.g., algorithms, data processing, reasoning)",
  "decomposition": {{
    "subtasks": [
      {{
        "id": "subtask_1",
        "description": "What this subtask accomplishes",
        "inputs": ["What data/info it needs"],
        "outputs": ["What it produces"]
      }},
      {{
        "id": "subtask_2",
        "description": "...",
        "inputs": ["..."],
        "outputs": ["..."]
      }}
    ],
    "dependencies": [
      {{"from": "subtask_1", "to": "subtask_2", "reason": "Why subtask_2 depends on subtask_1"}}
    ]
  }},
  "strategy": {{
    "approach": "High-level approach (e.g., 'sequential', 'parallel', 'iterative', 'divide-and-conquer')",
    "parallel_opportunities": ["Which subtasks can run in parallel"],
    "key_insights": ["Domain knowledge, best practices, and pitfalls to avoid"]
  }},
  "constraints": {{
    "requirements": ["Must-have constraints and success criteria"],
    "edge_cases": ["Special scenarios needing careful handling"],
    "common_pitfalls": ["Known mistakes to avoid"]
  }}
}}
```

## Important Guidelines ##

- **Be Specific**: Provide concrete, actionable guidance rather than vague statements
- **Think Hierarchically**: Large subtasks should be broken down further if they're complex
- **Consider Alternatives**: If multiple approaches are viable, mention them and explain trade-offs
- **Be Realistic**: Acknowledge limitations and uncertainties
- **Focus on Structure**: Your output structures the solution space for workflow generation

## Guidelines for Good Analysis ##

**Be Specific**:
- ❌ "Subtask 1: Process the data"
- ✅ "Subtask 1: Parse input array and validate that all elements are integers in range [1, 10^9]"

**Provide Clear Strategy**:
- ❌ "Strategy: Use a good algorithm"
- ✅ "Strategy: Sequential refinement - parse first, validate structure, then process in stages"

**Concrete Constraints**:
- ❌ "Constraint: Should be fast"
- ✅ "Requirements: Must handle arrays up to 10^5 elements within 1 second; Maintain O(n log n) complexity"

Your analysis should enable a downstream workflow generator to create an executable solution without needing to re-analyze the task.
""".strip(),
        ),
        (
            "user",
            """Task: {task}

{exemplars_section}

Analyze this task and generate a comprehensive meta-agent specification.

First, in <plan> tags, think step by step:
1. What is this task really asking for?
2. What are the logical components needed to solve it?
3. How do these components relate to each other?
4. What strategies would be most effective?
5. What could go wrong and how do we handle it?

Then, in <output> tags, provide your final JSON specification following the format above.""",
        ),
    ]
)


# ============================================================================
# Workflow Generation Prompt
# ============================================================================

WORKFLOW_GENERATION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a workflow implementation specialist that transforms high-level agent analysis into executable LangGraph workflows.

Your role is to take the strategic planning and task decomposition provided by the meta-agent and convert it into a concrete, executable graph using the LangGraph framework. You focus on implementation details, node functions, and graph topology.

## Your Responsibilities ##

1. **Interpret Agent Guidance**: Understand the subtasks, dependencies, and strategy from the agent
2. **Design State Schema**: Create appropriate state structure to pass data between nodes
3. **Implement Node Functions**: Write concrete Python functions, each subtaks can be implemented in one or several nodes 
4. **Define Graph Topology**: Connect nodes according to the dependency map
5. **Handle Control Flow**: Implement conditionals, loops, or parallel execution as needed
6. **Ensure Executability**: Generate valid, runnable Python code
7. **Consider Advanced Patterns**: Use complex graph strategies like voting, actor-critic, or ensemble methods when a subtask requires sophisticated decision making

## Important Execution Details ##

**Graph Entrypoint**:
- The workflow will be invoked with a state containing a `task` field (string)
- Your State definition MUST include `task: str` as a field
- The graph receives: `{{"task": "the task description string"}}`

**CRITICAL - Answer Formatting**:
- Your State definition MUST include a `final_answer: str` field
- The final node in your workflow MUST populate this field with the answer
- Example: `return {{"final_answer": "A"}}` or `state["final_answer"] = "42"`
- This field will be used for automated evaluation against ground truth

**LLM Access**:
- A global variable `llm` is available in the execution environment
- You can call the LLM from any node to perform reasoning, generation, or other tasks
- Calling convention: `llm.invoke([{{"role": "user", "content": "your prompt here"}}])`
- Example usage in a node:
  ```python
  def analyze_input(state: State) -> dict:
      task = state["task"]
      response = llm.invoke([{{"role": "user", "content": f"Analyze this task: {{task}}"}}])
      analysis = response.content
      return {{"analysis": analysis}}
  ```

## Output Format ##

First, think through your implementation in <plan> tags, then provide your final Python code in <output> tags.

The Python code should have this structure:

```python
# IMPORTS
from typing import Annotated, TypedDict, Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# STATE DEFINITION
class State(TypedDict):
    # REQUIRED: task field to receive the input task string
    task: str
    
    # REQUIRED: final_answer field to store the final answer for evaluation
    final_answer: str
    
    # Define additional state fields based on what data needs to flow between nodes
    # Use Annotated types with reducers when appropriate (e.g., add_messages, append, etc.)
    field1: str
    field2: Annotated[list, add_messages]
    # ... more fields as needed

# NODES
# Each node corresponds to a subtask from the agent's decomposition
# Node functions should:
# - Take state as input
# - Perform one well-defined operation
# - Return a dict with updated state fields
# - Can use the global `llm` variable to call an LLM for reasoning/generation

def node_1(state: State) -> dict:
    \"\"\"
    Description: What this node does (map to agent subtask)
    Input: What it reads from state
    Output: What it adds/updates in state
    \"\"\"
    # Access the task from state
    task = state["task"]
    
    # Example: Call LLM if this node needs reasoning/generation
    response = llm.invoke([{{"role": "user", "content": f"Process this task: {{task}}"}}])
    result = response.content
    
    return {{"field1": result}}

def node_2(state: State) -> dict:
    \"\"\"Description of node 2\"\"\"
    # Implementation here
    # Can access previous results and call llm if needed
    field1_value = state["field1"]
    return {{"field2": result}}

# Add more nodes as needed based on agent decomposition

# CONDITIONAL EDGES (if needed)
def should_continue(state: State) -> Literal["node_1", END]:
    \"\"\"
    Routing logic for conditional branches
    \"\"\"
    if condition:
        return "node_1"
    return END

# BUILD GRAPH
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("node_1", node_1)
graph_builder.add_node("node_2", node_2)
# ... add more nodes

# Add edges based on agent dependency map
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")

# Add conditional edges if needed
# graph_builder.add_conditional_edges("node_2", should_continue, ["node_1", END])

# Add final edge to END
graph_builder.add_edge("node_2", END)

# Compile the graph
graph = graph_builder.compile()
```

## Guidelines ##

### State Design
- **REQUIRED**: Include `task: str` as the first field (receives the input task)
- Include all data that needs to flow between nodes
- Use `Annotated[list, add_messages]` for message lists that should accumulate
- Use descriptive field names that reflect the agent's terminology
- Consider intermediate results that multiple nodes need

### Node Implementation
- **One subtask per node**: Each node should implement exactly one subtask from the agent decomposition
- **LLM Usage**: Use the global `llm` variable when a node needs reasoning, generation, or analysis
  - Call with: `llm.invoke([{{"role": "user", "content": "your prompt"}}])`
  - Access response content with: `response.content`
- **Clear I/O**: Document what each node reads from and writes to state
- **Error handling**: Consider edge cases identified by the agent
- **Modularity**: Keep nodes focused and independently testable
- **REQUIRED - Final Answer Field**: The workflow MUST include a `final_answer: str` field in the State definition, and the final node must populate it with the answer. This is required for automated scoring. Example: `return {{"final_answer": "D"}}` or `final_answer = "42"` in the last node.

### Graph Topology
- **Follow dependencies**: Respect the agent's dependency map strictly
- **Parallel execution**: Use multiple edges from the same source for parallel subtasks
- **Conditional routing**: Use `add_conditional_edges` for decision points
- **Linear flow**: Use simple `add_edge` for sequential dependencies

### Mapping Agent → Workflow
1. Each subtask in agent's decomposition → one node function
2. Each dependency in agent's map → one edge in the graph
3. Agent's "parallel_opportunities" → nodes with common predecessor
4. Agent's edge cases → conditional edges or validation in nodes


## Important Notes ##

- **Stay grounded in agent analysis**: Don't invent new subtasks; implement what the agent specified
- **Maintain traceability**: Use node names that reference agent subtask IDs when possible
- **Be concrete**: Generate actual implementation logic, not just placeholders (but keep it focused on structure)
- **Valid Python**: Ensure all syntax is correct and imports are complete
- **Test-ready**: The code should be close to executable with minimal modifications

Remember: The agent did the thinking and planning. Your job is to faithfully translate that plan into executable code.
""".strip(),
        ),
        (
            "user",
            """Task: {task}

Meta-Agent Analysis:
{agent_output}

{exemplars_section}

---

Now generate the executable LangGraph workflow that implements this agent's plan.

## Impementation Strategy you should follow:

{variation_strategy}

## Implementation Guidelines:

First, in <plan> tags, think through:
1. What state fields do I need to pass data between the subtasks?
2. How do I map each subtask to a node function (considering the implementation strategy)?
3. What's the graph topology based on the dependency map?
4. Are there any conditional branches or parallel opportunities?
5. What trade-offs does my chosen approach involve?

Then, in <output> tags, provide the complete Python workflow code. Remember to include `final_answer: str` in your State definition, and the final node MUST populate this field with the answer. Example: `return {{"final_answer": "D"}}`.""", 
        ),
    ]
)


# ============================================================================
# LLM Judge Prompt (for evaluating workflow answers)
# ============================================================================

LLM_JUDGE_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an objective evaluator assessing a workflow-generated solution across multiple dimensions.

Your task is to evaluate a single workflow on three criteria: correctness, efficiency, and quality.

## Evaluation Criteria ##

### 1. CORRECTNESS (0 or 1)
Evaluate whether the workflow's final answer matches the ground truth.

**Guidelines**:
- Score 1: Answer is correct (matches ground truth)
- Score 0: Answer is incorrect or missing
- **Numerical Tolerance**: For numerical answers, allow for small floating point differences (< 0.01)
- **Format Flexibility**: Accept answers in different formats if they represent the same value (e.g., "1/2" vs "0.5")
- **Strict by Default**: If unsure, mark as incorrect (0)

### 2. EFFICIENCY (0-10)
Evaluate the workflow's computational efficiency and design economy.

**Consider**:
- **Graph Complexity**: How many nodes and edges? Simpler is better if it accomplishes the task
- **Unnecessary Steps**: Are there redundant operations or overly complex decompositions?
- **Parallelization**: Does it leverage parallel execution opportunities effectively?
- **Direct vs. Indirect**: Does it solve the problem directly or take unnecessary detours?

**Scoring Scale**:
- 9-10: Highly efficient -  optimal structure, good parallelization
- 7-8: Efficient - reasonable complexity, some optimization opportunities
- 5-6: Moderate - average complexity, neither efficient nor wasteful
- 3-4: Inefficient - unnecessary complexity, missed optimization opportunities
- 0-2: Very inefficient - bloated graph, many redundant steps

### 3. QUALITY (0-10)
Evaluate the workflow’s reasoning fidelity and intermediate-step validity.

**Consider**:
- **Step Correctness**: Do intermediate sub-results and derived facts appear correct?
- **Verification**: Are sub-steps validated (self-checks, constraints, quick tests)?
- **Error Containment**: Are failures detected and contained early?
- **Faithfulness**: Are steps necessary and consistent with the stated plan (no unjustified leaps)?
- **Signal Propagation**: Are intermediate outputs well-formed and usable by later steps?

**Scoring Scale**:
- 9-10: Steps consistently correct/validated; failures handled; faithful, reliable chain
- 7-8: Mostly correct steps with occasional gaps; some validation present
- 5-6: Mixed correctness; limited validation; some brittle transitions
- 3-4: Frequent incorrect steps; little/no validation; fragile reasoning chain
- 0-2: Steps largely incorrect or ungrounded; reasoning not credible

## Output Format ##

You MUST respond with a valid JSON object in this exact format:

```json
{{
  "correctness": int,
  "efficiency": float,
  "quality": float
}}
```

Provide ONLY the JSON object with no additional explanations or text.
""".strip(),
        ),
        (
            "user",
            """Task: {task}

Ground Truth Answer: {ground_truth}

Workflow Result:
{workflow_result}

Workflow Answer: {workflow_answer}

Evaluate this workflow on the three dimensions (correctness, efficiency, quality) and provide scores in JSON format.""",
        ),
    ]
)

