#!/usr/bin/env python3
"""
Quick test script to verify the Agent-Workflow graph structure.
Run this to test the graph execution without full RL training.
Modified to use Qwen3-8B via Hugging Face Inference Providers.
"""

import os
import dotenv
from workflow_agent import AgentWorkflowGraph

dotenv.load_dotenv()

def test_graph_structure():
    """Test that the graph builds correctly and can execute."""
    print("=" * 80)
    print("Testing Agent-Workflow Graph Structure with Qwen3-8B (HuggingFace)")
    print("=" * 80)
    
    # Set up HuggingFace environment
    # Make sure HUGGINGFACEHUB_API_TOKEN is set in your .env file
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("\n⚠️  Warning: HUGGINGFACEHUB_API_TOKEN not found in environment.")
        print("Please set it in your .env file or export it:")
        print("export HUGGINGFACEHUB_API_TOKEN='hf_xxxxxxxxxxxxx'\n")
    
    # Configure to use HuggingFace with Qwen model
    os.environ["MODEL"] = "Qwen/Qwen3-8B"
    
    print(f"Using model: {os.environ['MODEL']}")
    print(f"Provider: HuggingFace Inference API\n")
    
    # Create graph with HuggingFace
    # The graph will auto-detect HuggingFace mode if HUGGINGFACEHUB_API_TOKEN is set
    graph = AgentWorkflowGraph( # type: ignore
        endpoint=None,  # Not needed for HuggingFace
        use_huggingface=True,  # Explicitly enable HuggingFace mode
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


if __name__ == "__main__":
    print("\n🚀 Agent-Workflow Graph Test Suite\n")
    
    # Test 2: Graph structure
    test1_passed = test_graph_structure()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Trainable Nodes Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    
    if test1_passed:
        print("\n🎉 All tests passed! The graph is ready for RL training.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

