import locale
import subprocess
from pathlib import Path


def run_solver_command(
    exec_path: str, arguments: list[str], timeout: int = 10, *, preserve_newlines: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a solver command, preserving stdout, stderr and the exit code.

    Set preserve_newlines for strict CLI format checks. Errors reach the caller.
    """
    result = subprocess.run(
        [exec_path, *arguments], capture_output=True, text=not preserve_newlines, timeout=timeout
    )
    if preserve_newlines:
        encoding = locale.getpreferredencoding(False)
        return subprocess.CompletedProcess(
            result.args, result.returncode,
            result.stdout.decode(encoding), result.stderr.decode(encoding),
        )
    return result


def run_solver_on_test(exec_path: str, onnx_path: Path, vnnlib_path: Path, timeout: int = 10) -> str:
    """
    Executes a VNNLIB 2.0 compatible solver on a single test case.
    
    The CLI standard is:
    <solver> verify <filepath> --network <name>=<filepath> [--timeout <seconds>]
    
    Returns the standard output of the solver.
    """
    cmd = [
        "verify",
        str(vnnlib_path),
        "--network", f"N={onnx_path}",
        "--timeout", str(timeout)
    ]
    
    try:
        result = run_solver_command(exec_path, cmd, timeout)
        output = result.stdout + "\n" + result.stderr        
        return output
        
    except subprocess.TimeoutExpired:
        return f"TimeoutExpired after {timeout} seconds"
    except Exception as e:
        return f"Error executing solver: {e}"
