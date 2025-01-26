"""
Event types + reference descriptions explaining their role.
"""

from dataclasses import dataclass


@dataclass
class HumanMsg:
    """
    A (normally non-automated) message from the user to the assistant. At the beginning of a conversation, that's the query containing the user's request. Later on, this may be a clarification, some advice, a question, etc.
    """
    text: str


@dataclass
class AssistantThought:
    """
    A thought from the assistant. This may help the user understand the assistant's intentions, but is mainly there to help the assistant solve problems and plan by giving it room to "think" step by step.

    It could be hidden from the user, and should not contain questions or requests for the user.

    A thought should always be followed by a another thought, or a more "decisive" interaction, like a code fragment, a message to the user, a completion signal... Thoughts are scaffolding.
    """
    text: str


@dataclass
class AssistantAction:
    """
    A brief statement explaining what the next action (code fragment) will do.
     """
    text: str


@dataclass
class CodeFragment:
    """
    A piece of Python code that the assistant requests to run in the REPL.
    """
    code: str


@dataclass
class AssistantMsg:
    """
    A message from the assistant to the user. The purpose of such a message is to directly address the user, by providing a final answer, requesting more information, etc.

    Messages of this type may generate notifications / draw the user's attention. They should be used sparingly, only when the assistant has something important to say or ask. In general, the assistant should stay as autonomous as possible.
    """
    text: str


@dataclass
class ExecutionResult:
    """
    The result of running a code fragment in the REPL.
    At first, just a string containing the (potentially truncated output)
    """
    output: str


@dataclass
class ResumeFrom:
    """
    Event to indicate that the conversation should be resumed from a specific event ID.
    Effectively rewinds the conversation history.
    """
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

# Somewhat duplicated, sadly.
individual_event_types = [
    HumanMsg,
    AssistantThought,
    AssistantAction,
    CodeFragment,
    AssistantMsg,
    ExecutionResult,
    ResumeFrom,
]
