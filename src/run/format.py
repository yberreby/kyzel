import re
from typing import Optional
from dataclasses import dataclass
from .execute import ExecutionResult

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
        # Remove ANSI color codes
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        text = re.sub(r'\x1b\][0-9;]*[a-zA-Z]', '', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove trailing whitespace while preserving empty lines
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        return text.rstrip() + '\n' if text else ''

    @classmethod
    def format_result(cls, result: ExecutionResult) -> LLMExecutionResult:
        """Convert raw execution result to LLM-friendly format."""
        outputs = []
        if result.output.stdout:
            outputs.append(cls.clean_text(result.output.stdout))
        if result.output.result is not None:
            outputs.append(cls.clean_text(str(result.output.result)))

        output = ''.join(outputs)

        # Format error to include exception type if available
        error = None
        if not result.success:
            if result.error:
                error_type = type(result.error).__name__
                error_msg = str(result.error)
                error = f"{error_type}: {error_msg}\n"
                if result.error_traceback:
                    error = cls.clean_text(result.error_traceback)
            elif result.error_traceback:
                error = cls.clean_text(result.error_traceback)

        return LLMExecutionResult(
            output=output,
            success=result.success,
            error=error
        )
