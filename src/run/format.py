import re
from typing import Optional
from dataclasses import dataclass
from .execute import ExecutionResult  # Using the unified ExecutionResult

@dataclass
class LLMExecutionResult:
    """Simplified, text-only execution result for LLM consumption."""
    output: str
    success: bool
    error: Optional[str] = None

class LLMFormatter:
    """Formats IPython execution results for LLM consumption."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text output."""
        if not text:
            return ''
        text = re.sub(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text_lines = [line.rstrip() for line in text.splitlines()]
        text = '\n'.join(text_lines).rstrip()
        return text + '\n' if text else ''

    @classmethod
    def format_result(cls, result: ExecutionResult) -> LLMExecutionResult: # Using the unified ExecutionResult
        """Convert raw execution result to LLM-friendly format."""
        outputs = []

        # Add stdout if present
        if result.output.stdout:
            outputs.append(cls.clean_text(result.output.stdout))

        # Add display output if present
        if result.output.display_output:
            outputs.append(cls.clean_text(result.output.display_output))

        # Always add result if it exists - this ensures expression values are included
        # if result.output.result is not None:
        #     outputs.append(cls.clean_text(repr(result.output.result)))

        output = ''.join(outputs)

        error = None
        if not result.success:
            if result.error_traceback:
                error = cls.clean_text(result.error_traceback)
                if result.error:
                    error_type = type(result.error).__name__
                    error_msg = str(result.error)
                    if error.strip():
                        error = f"{error_type}: {error_msg}\n{error}"
                    else:
                        error = f"{error_type}: {error_msg}\n"
            elif result.error:
                error_type = type(result.error).__name__
                error_msg = str(result.error)
                error = f"{error_type}: {error_msg}\n"
            elif result.error_traceback:
                error = cls.clean_text(result.error_traceback)

        return LLMExecutionResult(
            output=output,
            success=result.success,
            error=error
        )
