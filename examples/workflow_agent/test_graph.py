#!/usr/bin/env python3
"""
Quick test script to verify the Agent-Workflow graph structure.
Run this to test the graph execution without full RL training.
"""

import os
import dotenv
from workflow_agent import AgentWorkflowGraph

dotenv.load_dotenv()

def test_graph_structure():
    """Test that the graph builds correctly and can execute."""
    print("=" * 80)
    print("Testing Agent-Workflow Graph Structure")
    print("=" * 80)
    
    # Create graph
    graph = AgentWorkflowGraph( # type: ignore
        endpoint=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        debug=True,
    ).build_graph()
    
    print("\n✓ Graph built successfully!")
    print(f"\nGraph nodes: {list(graph.nodes.keys())}")
    
    # Test execution with a simple task
    print("\n" + "=" * 80)
    print("Testing Graph Execution")
    print("=" * 80)
    
    test_task = {
        "task": "Write a Python function to calculate the Fibonacci sequence up to N numbers."
    }
    
    print(f"\nTask: {test_task['task']}")
    print("\nExecuting graph...")
    
    try:
        result = graph.invoke(test_task) # type: ignore
        
        print("\n✓ Graph executed successfully!")
        print(f"\nAgent Output: {result['agent_output'][:200]}...")
        print(f"\nWorkflow 1: {result['workflow_1'][:150]}..." if result['workflow_1'] else "None")
        print(f"Workflow 2: {result['workflow_2'][:150]}..." if result['workflow_2'] else "None")
        print(f"Workflow 3: {result['workflow_3'][:150]}..." if result['workflow_3'] else "None")
        print(f"\nFinal Score: {result['final_score']:.3f}")
        print(f"Best Workflow: {result['best_workflow'][:150]}..." if result['best_workflow'] else "None")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainable_nodes():
    """Test that trainable nodes are correctly identified."""
    import re
    
    print("\n" + "=" * 80)
    print("Testing Trainable Node Pattern")
    print("=" * 80)
    
    pattern = r"generate_(agent|workflow_\d+)"
    
    nodes = [
        "generate_agent",
        "generate_workflow_1",
        "generate_workflow_2",
        "generate_workflow_3",
        "execute_workflow_1",
        "execute_workflow_2",
        "execute_workflow_3",
        "aggregate_results",
    ]
    
    print(f"\nPattern: {pattern}")
    print("\nNode Classification:")
    
    for node in nodes:
        is_trainable = bool(re.search(pattern, node))
        status = "✓ TRAINABLE" if is_trainable else "✗ Non-trainable"
        print(f"  {node:30s} {status}")
    
    # Verify correct classification
    trainable_count = sum(1 for node in nodes if re.search(pattern, node))
    expected_trainable = 4  # agent + 3 workflows
    
    if trainable_count == expected_trainable:
        print(f"\n✓ Correct: {trainable_count}/{len(nodes)} nodes are trainable")
        return True
    else:
        print(f"\n✗ Error: Expected {expected_trainable} trainable nodes, got {trainable_count}")
        return False


if __name__ == "__main__":
    print("\n🚀 Agent-Workflow Graph Test Suite\n")
    
    # Test 1: Trainable nodes
    test1_passed = test_trainable_nodes()
    
    # Test 2: Graph structure
    test2_passed = test_graph_structure()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Trainable Nodes Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Graph Execution Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The graph is ready for RL training.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

