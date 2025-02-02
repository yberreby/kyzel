"""
At any given point, a session can be flattened into a list of ChatML messages,
applying any appropriate rewriting transformations.
"""

from typing import List
import traceback
from typing import Optional
from src.types.chatml import Conversation, Msg as ChatMLMsg, NormalRole
from src.types import (
    Session,
    AssistantAction,
    EventBody,
    HumanMsg,
    AssistantMsg,
    AssistantThought,
    CodeFragment,
    ResumeFrom,
    SessionEvent,
    ExecutionResult,
)
from src.run.format import LLMExecutionResult, LLMFormatter
from src.postproc import parse_constrained_message
from src.prompts import DEFAULT_SYSTEM_PROMPT


def event_source_role(event_body: EventBody) -> NormalRole:
    match event_body:
        case HumanMsg():
            return "user"
        case AssistantMsg() | AssistantThought() | AssistantAction():
            return "assistant"
        case CodeFragment():
            return "assistant"
        case ExecutionResult():
            return "user"
        case _:
            raise ValueError(f"No role mapped to type for event: {event_body}")


def as_code_fences(code: str) -> str:
    return f"\n```python\n{code}\n```\n"


def as_thought_block(text: str) -> str:
    return f"<thought>{text}</thought>"


def as_action_block(action: str) -> str:
    return f"<action>{action}</action>"


def as_output_block(res: LLMExecutionResult) -> str:
    inner = res.to_plaintext()
    return f"<output>{inner}</output>"

def as_error_block(result: str) -> str:
    return f"<error>{result}</error>"


def event_to_plaintext(event_body: EventBody) -> str:
    """
    Convert event body to plaintext for LLM input.
    """
    match event_body:
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
        case ExecutionResult():
            fmt_res = LLMFormatter.format_result( event_body )
            return as_output_block(fmt_res)
        case _:
            raise ValueError(f"No plaintext mapped to type for event: {event_body}")


def ensure_consistency(conv: Conversation):
    """Ensure no two messages of the same role are in a row."""
    for i in range(1, len(conv)):
        if conv[i]["role"] == conv[i - 1]["role"]:
            raise ValueError(
                f"Two messages of same role in a row: {conv[i - 1]} and {conv[i]}, in conversation: {conv}"
            )


def validate_flattened_assistant_msg(msg: ChatMLMsg):
    """Validate assistant message structure."""
    try:
        parse_constrained_message(msg["content"])
    except Exception as e:
        traceback.print_exc()
        raise ValueError(
            f"Assistant message validation failed:\nContent: {msg['content']}\nError: {str(e)}"
        )


def session_to_chatml(session: Session) -> Conversation:
    """Convert session events to ChatML conversation format."""
    system_msg: ChatMLMsg = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    conv: Conversation = [system_msg]

    prev_role: Optional[NormalRole] = None
    cut_off_index = None

    for i, session_event in enumerate(session.events):
        event = session_event.body
        if isinstance(event, ResumeFrom):
            cut_off_index = _find_event_index_by_id(session.events, event.from_event_id)
            if cut_off_index is not None:
                print(
                    f"ResumeFrom found, cutting history after event id: {event.from_event_id} at index {cut_off_index}"
                )
                session.events = session.events[: cut_off_index + 1]
                break
            else:
                print(
                    f"Warning: ResumeFrom refers to unknown event id: {event.from_event_id}. Ignoring."
                )
                continue

    for session_event in session.events:
        event = session_event.body
        role = event_source_role(event)
        new_text: str = event_to_plaintext(event)

        if prev_role == role:
            prev_msg: ChatMLMsg = conv[-1]
            prev_msg["content"] += "\n" + new_text
            continue

        msg: ChatMLMsg = {"role": role, "content": new_text}
        prev_role = role
        conv.append(msg)

    ensure_consistency(conv)

    for msg in conv:
        if msg["role"] == "assistant":
            validate_flattened_assistant_msg(msg)

    return conv


def _find_event_index_by_id(
    session_events: List[SessionEvent], event_id: str
) -> Optional[int]:
    """Find event index by ID."""
    for index, session_event in enumerate(session_events):
        if session_event.event_id == event_id:
            return index
    return None
