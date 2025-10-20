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


    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
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
        "task": """A reaction of a liquid organic compound, which molecules consist of carbon and hydrogen atoms, is performed at 80 centigrade and 20 bar for 24 hours. In the proton nuclear magnetic resonance spectrum, the signals with the highest chemical shift of the reactant are replaced by a signal of the product that is observed about three to four units downfield.
Compounds from which position in the periodic system of the elements, which are also used in the corresponding large-scale industrial process, have been mostly likely initially added in small amounts?
A) A metal compound from the fifth period.
B) A metal compound from the fifth period and a non-metal compound from the third period.
C) A metal compound from the fourth period.
D) A metal compound from the fourth period and a non-metal compound from the second period.""",
        "ground_truth": "D"
    }
    
    print(f"\nTask: {test_task['task']}")
    print("\nExecuting graph...")
    
    try:
        result = graph.invoke(test_task) # type: ignore
        
        print("\n✓ Graph executed successfully!")
        
        # Save agent output to file
        with open("agent_output.txt", "w") as f:
            f.write(result['agent_output'])
            
        # Save each workflow output to separate files
        for i in range(1, 4):
            workflow_key = f'workflow_{i}'
            if result.get(workflow_key):
                with open(f"workflow_{i}_output.txt", "w") as f:
                    f.write(f"Workflow {i} Output:\n")
                    f.write(result[workflow_key])
                    f.write(f"\n\nWorkflow {i} Result:\n")
                    f.write(str(result.get(f'{workflow_key}_result', 'No result')))
                    f.write(f"\n\nWorkflow {i} Score: {result.get(f'{workflow_key}_score', 0.0):.3f}")
                    f.write(f"\nGround Truth: {result.get('ground_truth', 'N/A')}")
        
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

