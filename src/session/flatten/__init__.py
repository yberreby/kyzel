"""
At any given point, a session can be flattened into a list of ChatML messages,
applying any appropriate rewriting transformations.
"""

from typing import Optional
from src.chatml import Conversation, Msg as ChatMLMsg, NormalRole
from src.session import Session
from src.session.event.types import (
    AssistantAction,
    EventBody,
    HumanMsg,
    AssistantMsg,
    AssistantThought,
    CodeFragment,
    ExecutionResult,
)


def event_source_role(event: EventBody) -> NormalRole:
    match event:
        # These are rather clear-cut.
        case HumanMsg():
            return "user"
        case AssistantMsg() | AssistantThought() | AssistantAction():
            return "assistant"
        # Code fragments are always emitted by the assistant (at least at the moment).
        case CodeFragment():
            return "assistant"
        # Execution results are fed through user messages, not system.
        case ExecutionResult():
            return "user"
        case _:
            raise ValueError(f"No role mapped to type for event: {event}")


def as_code_fences(code: str) -> str:
    # Models usually love Markdown and code blocks.
    # Could evaluate whether 'py' is more common...
    # Empirically, it seems that Phi4 likes 'python' out of the box.
    # These newlines are obviously IMPORTANT.
    return f"```python\n{code}\n```"


def as_thought_block(text: str) -> str:
    # Might as well be close to what DeepSeek is doing.
    # You never know. Maybe we'll want to use distills...
    return f"<thought>{text}</thought>"


def as_action_block(action: str) -> str:
    # This is succinct so maybe could just have a prefix... but let's try to be consistent.
    return f"<action>{action}</output>"


def as_output_block(result: str) -> str:
    # Just a simple XML block for now.
    # Could just be <output/> when empty - but let's not confuse the models.
    return f"<output>{result}</output>"


def event_to_plaintext(event: EventBody) -> str:
    """
    Right now, we have no rewriting, and we can have a 1-1 mapping between events and a plaintext representation that is model-friendly.
    """
    match event:
        case HumanMsg(text):
            return text
        case AssistantMsg(text):
            return text
        case AssistantThought(text):
            return as_thought_block(text)
        case AssistantAction(text):
            return as_action_block(text)
        case CodeFragment(code):
            return as_code_fences(code)
        case ExecutionResult(result):
            return as_output_block(result)
        case _:
            raise ValueError(f"No plaintext mapped to type for event: {event}")


def ensure_consistency(conv: Conversation):
    """
    A correct conversation should not have two messages of the same role in a row.
    """
    for i in range(1, len(conv)):
        if conv[i]["role"] == conv[i - 1]["role"]:
            raise ValueError(
                f"Found two messages of the same role in a row: {conv[i - 1]} and {conv[i]}, in conversation: {conv}"
            )


def flatten_to_chatml(session: Session) -> Conversation:
    """
    For now, no deletion or anything fancy.
    Just coalescing of relevant contiguous messages.
    No system prompt.
    """
    conv: Conversation = []
    prev_role: Optional[NormalRole] = None
    for event in session.events:
        role = event_source_role(event)
        new_text: str = event_to_plaintext(event)

        # If the role is the same as the previous message, we can coalesce.
        if prev_role == role:
            # There must be a previous message, otherwise there would be no previous role.
            prev_msg: ChatMLMsg = conv[-1]
            # Just newline-separated for now, dead simple.
            prev_msg["content"] += "\n" + new_text
            continue

        # Standard case: new message.
        msg: ChatMLMsg = {"role": role, "content": new_text}
        prev_role = role
        conv.append(msg)

    # Cheap sanity check.
    ensure_consistency(conv)

    return conv
