import subprocess
from pathlib import Path


def run_solver_on_test(exec_path: str, onnx_path: Path, vnnlib_path: Path, timeout: int = 10) -> str:
    """
    Executes a VNNLIB 2.0 compatible solver on a single test case.
    
    The CLI standard is:
    <solver> verify <filepath> --network <name>=<filepath> [--timeout <seconds>]
    
    Returns the standard output of the solver.
    """
    cmd = [
        exec_path,
        "verify",
        str(vnnlib_path),
        "--network", f"N={onnx_path}",
        "--timeout", str(timeout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + "\n" + result.stderr        
        return output
        
    except subprocess.TimeoutExpired:
        return f"TimeoutExpired after {timeout} seconds"
    except Exception as e:
        return f"Error executing solver: {e}"
