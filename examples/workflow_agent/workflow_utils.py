import json
from typing import Any, Dict, Optional

def extract_output_tag(content: str) -> str:
    """Extract content from <output> tags if present, otherwise return full content."""
    import re
    
    # Try to find content within <output> tags
    output_match = re.search(r'<output>\s*(.*?)\s*</output>', content, re.DOTALL | re.IGNORECASE)
    if output_match:
        return output_match.group(1)
    
    return content

def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response (looks in <output> tags first)."""
    import re
    
    # First, extract content from <output> tags if present
    output_content = extract_output_tag(content)
    
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
        return None

def parse_python_code(content: str) -> Optional[str]:
    """Extract Python code from LLM response (looks in <output> tags first)."""
    import re
    
    # First, extract content from <output> tags if present
    output_content = extract_output_tag(content)
    
    # Try to find Python code block
    python_match = re.search(r'```python\s*(.*?)\s*```', output_content, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # If no code block found, return None
    return None

def parse_judge_scores(content: str) -> Dict[str, float]:
    """
    Parse structured judge response with correctness, efficiency, and quality scores.
    
    Returns:
        Dict with keys 'correctness' (0 or 1), 'efficiency' (0-10), 'quality' (0-10)
        Defaults to zeros if parsing fails.
    """
    scores_dict = parse_json_response(content) # type: ignore
    
    if scores_dict is None:
        return {"correctness": 0.0, "efficiency": 0.0, "quality": 0.0}
    
    # Extract and validate scores
    correctness = float(scores_dict.get("correctness", 0))
    efficiency = float(scores_dict.get("efficiency", 0))
    quality = float(scores_dict.get("quality", 0))
    
    # Clamp values to valid ranges
    correctness = max(0.0, min(1.0, correctness))
    efficiency = max(0.0, min(10.0, efficiency))
    quality = max(0.0, min(10.0, quality))
    
    return {
        "correctness": correctness,
        "efficiency": efficiency,
        "quality": quality,
    }

def compute_composite_score(correctness: float, efficiency: float, quality: float) -> float:
    """
    Compute composite score from multi-dimensional metrics.
    
    Formula: 0.5 * correctness + 0.2 * (efficiency/10) + 0.3 * (quality/10)
    
    Args:
        correctness: Binary score (0 or 1)
        efficiency: Score from 0-10
        quality: Score from 0-10
        
    Returns:
        Composite score in range [0.0, 1.0]
    """
    return 0.5 * correctness + 0.2 * (efficiency / 10.0) + 0.3 * (quality / 10.0)