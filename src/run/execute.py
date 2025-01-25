from IPython import get_ipython
from IPython.utils.capture import capture_output
from typing import Optional
from dataclasses import dataclass
import traceback

@dataclass
class CellOutput:
    """Raw output from cell execution."""
    stdout: str
    stderr: str
    result: Optional[object] = None

class ExecutionResult:
    """Complete result of cell execution with rich error information."""
    def __init__(self,
                 output: CellOutput,
                 success: bool,
                 error: Optional[Exception] = None,
                 error_traceback: Optional[str] = None):
        self.output = output
        self.success = success
        self.error = error  # The actual exception object
        self.error_traceback = error_traceback  # Raw traceback string

class IPythonExecutor:
    """Raw IPython cell execution with minimal processing."""

    def __init__(self):
        self.ipython = get_ipython()
        if self.ipython is None:
            raise RuntimeError("Not running in IPython environment")

    def execute(self, code: str) -> ExecutionResult:
        """Execute code and return raw results."""
        try:
            with capture_output(display=True) as captured:
                result = self.ipython.run_cell(code, silent=True)

                output = CellOutput(
                    stdout=captured.stdout or '',
                    stderr=captured.stderr or '',
                    result=result.result
                )

                if not result.success:
                    error = None
                    if result.error_in_exec:
                        error = result.error_in_exec
                    elif result.error_before_exec:
                        error = result.error_before_exec

                    # Get the traceback if we have an error
                    tb = None
                    if error:
                        tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
                    elif captured.stderr:
                        tb = captured.stderr

                    return ExecutionResult(
                        output=output,
                        success=False,
                        error=error,
                        error_traceback=tb
                    )

                return ExecutionResult(
                    output=output,
                    success=True
                )

        except Exception as e:
            tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            return ExecutionResult(
                output=CellOutput('', '', None),
                success=False,
                error=e,
                error_traceback=tb
            )
