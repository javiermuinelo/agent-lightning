#!/usr/bin/env python3
"""
Token counter for Agent-Workflow prompts using Qwen3-8B tokenizer.

This script loads the prompt templates and counts how many tokens each template uses.
"""

from transformers import AutoTokenizer

from prompts import (
    AGENT_GENERATION_PROMPT,
    WORKFLOW_GENERATION_PROMPT,
    WORKFLOW_VARIATION_STRATEGIES,
)


def count_template_tokens(tokenizer, prompt_template):
    """
    Count tokens in a prompt template (without filling in variables).
    
    Args:
        tokenizer: The tokenizer to use
        prompt_template: ChatPromptTemplate to analyze
    
    Returns:
        Token count per message and total
    """
    messages = []
    total_tokens = 0
    
    for message_prompt in prompt_template.messages:
        # Extract role from the message prompt template
        role = message_prompt.__class__.__name__.replace('MessagePromptTemplate', '').lower()
        
        # Extract the template string content
        if hasattr(message_prompt.prompt, 'template'):
            content = message_prompt.prompt.template
        elif hasattr(message_prompt, 'prompt'):
            content = str(message_prompt.prompt)
        else:
            content = str(message_prompt)
        
        # Count tokens
        tokens = tokenizer.encode(content, add_special_tokens=False)
        num_tokens = len(tokens)
        total_tokens += num_tokens
        
        messages.append({
            'role': role,
            'content_length': len(content),
            'tokens': num_tokens,
            'preview': content[:150].replace('\n', ' ') + '...' if len(content) > 150 else content.replace('\n', ' ')
        })
    
    return total_tokens, messages


def main():
    print("=" * 80)
    print("AGENT-WORKFLOW PROMPT TEMPLATE TOKEN ANALYSIS")
    print("=" * 80)
    print()
    
    # Load Qwen3-8B tokenizer
    print("Loading Qwen2.5-7B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print(f"✓ Loaded tokenizer: {tokenizer.__class__.__name__}")
    print()
    
    # ========================================================================
    # 1. Agent Generation Prompt Template
    # ========================================================================
    
    agent_tokens, agent_messages = count_template_tokens(
        tokenizer,
        AGENT_GENERATION_PROMPT
    )
    
    # ========================================================================
    # 2. Workflow Generation Prompt Template
    # ========================================================================
    
    workflow_tokens, workflow_messages = count_template_tokens(
        tokenizer,
        WORKFLOW_GENERATION_PROMPT
    )
    
    
    # ========================================================================
    # 3. Variation Strategies
    # ========================================================================
    strategy_tokens = {}
    for strategy_id, strategy_text in WORKFLOW_VARIATION_STRATEGIES.items():
        tokens = tokenizer.encode(strategy_text, add_special_tokens=False)
        num_tokens = len(tokens)
        strategy_tokens[strategy_id] = num_tokens
    
    avg_strategy_tokens = sum(strategy_tokens.values()) / len(strategy_tokens)
    
    # ========================================================================
    # 4. Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Template Token Counts:")
    print("-" * 40)
    print(f"Agent Generation Template:        {agent_tokens:>6,} tokens")
    print(f"Workflow Generation Template:     {workflow_tokens:>6,} tokens")
    print(f"Avg Variation Strategy:           {avg_strategy_tokens:>6,.0f} tokens")
    print()
    
    # Calculate effective prompt sizes
    print("Effective Prompt Sizes (template + variation):")
    print("-" * 40)
    print(f"Agent Prompt (fixed):             {agent_tokens:>6,} tokens")
    for strategy_id in [1, 2, 3]:
        total = workflow_tokens + strategy_tokens[strategy_id]
        print(f"Workflow Prompt (variation {strategy_id}):   {total:>6,} tokens")
    
    print("Note: Actual token usage will be higher when including:")
    print("  - User task description")
    print("  - Agent's generated JSON output (passed to workflow)")
    print("  - Model's generated responses")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

