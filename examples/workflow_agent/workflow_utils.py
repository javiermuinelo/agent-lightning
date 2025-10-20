import json
from typing import Any, Dict, Optional
import ast
import io
import multiprocessing as mp
from contextlib import redirect_stdout
import resource  

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


# ============================
# Safe Python Execution Utils
# ============================

_DISALLOWED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Attribute,  # prevents obj.__dict__ style escapes in most cases
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)


def sanitize_python_code(code: str) -> None:
    """Parse and validate Python code disallowing dangerous constructs.

    Raises a ValueError if disallowed nodes are detected.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:  # pragma: no cover
        raise ValueError(f"Invalid Python syntax: {e}")

    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_AST_NODES):
            raise ValueError(f"Disallowed Python construct: {type(node).__name__}")
        # Disallow calling __import__ directly via Name
        if isinstance(node, ast.Name) and node.id == "__import__":
            raise ValueError("Use of __import__ is disallowed")


def _executor_worker(code: str, input_vars: Dict[str, Any] | None, memory_limit_mb: int, queue: "mp.Queue[Dict[str, Any]]") -> None:
    """Worker to execute code in a restricted environment and send back results."""
    # Apply resource limits (if available and on POSIX)
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        max_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
    except Exception:
        pass

    try:
        # Limit CPU time to prevent runaway code (2 sec hard limit)
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    except Exception:
        pass

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "range": range,
        "len": len,
        "enumerate": enumerate,
        "sorted": sorted,
        "all": all,
        "any": any,
        "round": round,
    }

    globals_dict = {
        "__builtins__": safe_builtins,
    }

    locals_dict: Dict[str, Any] = {}
    if input_vars:
        # Ensure we don't allow overriding globals/builtins keys
        for k, v in input_vars.items():
            if k in ("__builtins__",):
                continue
            locals_dict[k] = v

    stdout_capture = io.StringIO()
    error: str | None = None
    result_value: Any = None

    try:
        with redirect_stdout(stdout_capture):
            exec(code, globals_dict, locals_dict)
        # Convention: if snippet assigns a variable named 'result', return it
        result_value = locals_dict.get("result")
    except Exception as e:  # pragma: no cover
        error = f"{type(e).__name__}: {e}"

    queue.put({
        "result": result_value,
        "stdout": stdout_capture.getvalue(),
        "error": error,
    })


def execute_python_code(
    code: str,
    input_vars: Dict[str, Any] | None = None,
    timeout_s: float = 2.0,
    memory_limit_mb: int = 128,
) -> Dict[str, Any]:
    """Execute sanitized Python code in a separate process with basic sandboxing.

    Returns a dict with keys: result, stdout, error.
    The snippet can set a variable named `result` to pass back a value.
    """
    sanitize_python_code(code)

    ctx = mp.get_context("spawn")
    queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()  # type: ignore
    proc = ctx.Process(target=_executor_worker, args=(code, input_vars or {}, memory_limit_mb, queue))
    proc.daemon = True
    proc.start()

    proc.join(timeout_s)
    if proc.is_alive():
        try:
            proc.terminate()
        finally:
            proc.join(0.1)
        return {"result": None, "stdout": "", "error": f"Timeout after {timeout_s}s"}

    try:
        payload = queue.get_nowait()
    except Exception:  # pragma: no cover
        payload = {"result": None, "stdout": "", "error": "No output captured"}

    # Ensure shape
    return {
        "result": payload.get("result"),
        "stdout": payload.get("stdout", ""),
        "error": payload.get("error"),
    }