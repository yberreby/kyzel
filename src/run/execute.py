import traceback
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
    display_output: str
    result: Optional[object] = None


# FIXME: this type and the logic surrounding it need a rework.
@dataclass
class ExecutionResult:
    """
    The result of running a code fragment in the REPL.
    At first, just a string containing the (potentially truncated output)
    """
    output: CellOutput
    success: bool
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None

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
                cell_result = self.ipython.run_cell(code)

                stdout = captured.stdout or ''
                stderr = captured.stderr or ''
                display_output = ''
                captured_result = None # Initialize captured_result

                if hasattr(captured, '_outputs'):
                    for output in captured._outputs:
                        if isinstance(output, dict):
                            if 'text/plain' in output['data']:
                                display_output += str(output['data']['text/plain']) + '\n'
                                # IMPORTANT: When using capture_output(display=True), IPython redirects
                                # the value of naked trailing expressions (like '1+1') to `captured._outputs`
                                # as a 'display output' of type 'text/plain', NOT to `cell_result.result`.
                                # We capture the *first* 'text/plain' output as the effective "result"
                                # of the cell for naked trailing expressions. This is crucial for capturing
                                # the output of simple expressions like '1+1' or variable values without explicit print().
                                if captured_result is None: # Capture only the first 'text/plain' output as result.
                                    captured_result = output['data']['text/plain']


                output = CellOutput(
                    stdout=stdout,
                    stderr=stderr,
                    display_output=display_output,
                    result=captured_result # Use captured_result from _outputs as cell result
                )

                if not cell_result.success:
                    error = None
                    tb = None
                    if cell_result.error_in_exec:
                        error = cell_result.error_in_exec
                    elif cell_result.error_before_exec:
                        error = cell_result.error_before_exec

                    if error:
                        tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
                    elif captured.stderr:
                        tb = captured.stderr
                    else:
                        tb = "Cell execution failed without detailed error info."

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
                output=CellOutput('', '', '', None),
                success=False,
                error=e,
                error_traceback=tb
            )
