"""
tools/code_runner.py
--------------------
Sandboxed Python execution engine for the ExecutionEngineerAgent.

Responsibilities:
  - Pre-inject standard data-science imports so the LLM doesn't waste tokens on boilerplate.
  - Execute agent-generated code in an isolated subprocess with a configurable timeout.
  - Return a rich result dict: stdout, stderr, success flag, and a list of every image
    artifact written to disk during execution.
  - Perform clean teardown of the temporary script file regardless of outcome.
"""

import glob
import os
import subprocess
import sys

# Standard preamble injected before every agent-generated script block.
# The CSV path is interpolated at call-time so the LLM never needs to know the path.
_PREAMBLE_TEMPLATE = """\
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"{csv_path}")

"""

_TEMP_SCRIPT = "temp_analysis_script.py"


def execute_analysis_code(
    script_contents: str,
    csv_path: str,
    timeout: int = 90,
) -> dict:
    """
    Execute an agent-generated Python script block inside a subprocess sandbox.

    Parameters
    ----------
    script_contents : str
        Raw Python code produced by the ExecutionEngineerAgent (no preamble needed).
    csv_path : str
        Absolute or relative path to the churn CSV file.
    timeout : int
        Maximum wall-clock seconds to allow for the script to complete.

    Returns
    -------
    dict with keys:
        success      : bool
        output       : str   — captured stdout
        error        : str   — captured stderr (populated on failure too)
        artifacts    : list[str]  — paths of every .png written during execution
        returncode   : int
    """
    full_code = _PREAMBLE_TEMPLATE.format(csv_path=csv_path) + script_contents

    # Snapshot of PNG files on disk *before* execution so we can diff afterwards.
    pre_run_pngs = set(glob.glob("*.png"))

    try:
        with open(_TEMP_SCRIPT, "w", encoding="utf-8") as fh:
            fh.write(full_code)

        result = subprocess.run(
            [sys.executable, _TEMP_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _cleanup()
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {timeout} seconds.",
            "artifacts": [],
            "returncode": -1,
        }
    except Exception as exc:
        _cleanup()
        return {
            "success": False,
            "output": "",
            "error": f"Subprocess launch error: {exc}",
            "artifacts": [],
            "returncode": -1,
        }
    finally:
        _cleanup()

    # Determine which PNG files were *newly created* during this run.
    post_run_pngs = set(glob.glob("*.png"))
    new_artifacts = sorted(post_run_pngs - pre_run_pngs)

    if result.returncode != 0:
        return {
            "success": False,
            "output": result.stdout,
            "error": result.stderr,
            "artifacts": new_artifacts,
            "returncode": result.returncode,
        }

    return {
        "success": True,
        "output": result.stdout,
        "error": result.stderr,        # may contain warnings even on success
        "artifacts": new_artifacts,
        "returncode": result.returncode,
    }


def _cleanup():
    """Remove the temporary script file if it still exists."""
    if os.path.exists(_TEMP_SCRIPT):
        try:
            os.remove(_TEMP_SCRIPT)
        except OSError:
            pass  # Non-critical; don't crash the orchestrator.