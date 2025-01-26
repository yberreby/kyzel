"""
Event types + reference descriptions explaining their role.
"""

from dataclasses import dataclass
from src.run.execute import ExecutionResult


@dataclass
class HumanMsg:
    """User message."""

    text: str


@dataclass
class AssistantThought:
    """Assistant's thought process."""

    text: str


@dataclass
class AssistantAction:
    """Assistant's intended action (code execution)."""

    text: str


@dataclass
class CodeFragment:
    """Code to be executed."""

    code: str


@dataclass
class AssistantMsg:
    """Message from assistant to user."""

    text: str


@dataclass
class ResumeFrom:
    """Resumes conversation from a prior event."""

    from_event_id: str


EventBody = (
    HumanMsg
    | AssistantThought
    | AssistantAction
    | CodeFragment
    | AssistantMsg
    | ExecutionResult
    | ResumeFrom
)

individual_event_types = [
    HumanMsg,
    AssistantThought,
    AssistantAction,
    CodeFragment,
    AssistantMsg,
    ExecutionResult,
    ResumeFrom,
]
