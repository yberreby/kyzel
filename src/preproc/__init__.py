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
    ExecutionResult,
    ResumeFrom, # Import ResumeFrom
    SessionEvent, # Import SessionEvent
)
from src.postproc import parse_constrained_message


def event_source_role(event_body: EventBody) -> NormalRole: # Expects EventBody now
    match event_body: # Matching on event_body
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
            raise ValueError(f"No role mapped to type for event: {event_body}") # Reflect type of event_body


def as_code_fences(code: str) -> str:
    # Models usually love Markdown and code blocks.
    # Could evaluate whether 'py' is more common...
    # Empirically, it seems that Phi4 likes 'python' out of the box.
    # These newlines are obviously IMPORTANT.
    return f"\n```python\n{code}\n```\n"


def as_thought_block(text: str) -> str:
    # Might as well be close to what DeepSeek is doing.
    # You never know. Maybe we'll want to use distills...
    return f"<thought>{text}</thought>"


def as_action_block(action: str) -> str:
    # This is succinct so maybe could just have a prefix... but let's try to be consistent.
    return f"<action>{action}</action>"


def as_output_block(result: str) -> str:
    # Just a simple XML block for now.
    # Could just be <output/> when empty - but let's not confuse the models.
    return f"<output>{result}</output>"


def event_to_plaintext(event_body: EventBody) -> str: # Expects EventBody now
    """
    Right now, we have no rewriting, and we can have a 1-1 mapping between events and a plaintext representation that is model-friendly.
    """
    match event_body: # Matching on event_body
        case HumanMsg(text): # Access text attribute directly
            return text
        case AssistantMsg(text): # Access text attribute directly
            return text
        case AssistantThought(text): # Access text attribute directly
            return as_thought_block(text)
        case AssistantAction(text): # Access text attribute directly
            return as_action_block(text)
        case CodeFragment(code): # Access code attribute directly
            return as_code_fences(code)
        case ExecutionResult(output): # Access output attribute directly
            return as_output_block(output)
        case _:
            raise ValueError(f"No plaintext mapped to type for event: {event_body}") # Reflect type of event_body


def ensure_consistency(conv: Conversation):
    """
    A correct conversation should not have two messages of the same role in a row.
    """
    for i in range(1, len(conv)):
        if conv[i]["role"] == conv[i - 1]["role"]:
            raise ValueError(
                f"Found two messages of the same role in a row: {conv[i - 1]} and {conv[i]}, in conversation: {conv}"
            )

def validate_flattened_assistant_msg(msg: ChatMLMsg):
    """
    Ensure that an assistant message conforms to our expected structure.
    Raises ValueError with details if validation fails.
    """

    try:
        # If this succeeds, the message looks valid
        parse_constrained_message(msg["content"])
    except Exception as e:
        # Get detailed message including the problematic content
        traceback.print_exc()
        raise ValueError(
            f"Assistant message failed validation:\n"
            f"Content: {msg['content']}\n"
            f"Error: {str(e)}"
        )

def session_to_chatml(session: Session) -> Conversation:
    """
    For now, no deletion or anything fancy.
    Just coalescing of relevant contiguous messages.
    No system prompt.

    Applies history rewriting based on ResumeFrom.
    Enforces parser-based validation of assistant messages.
    """
    conv: Conversation = []
    prev_role: Optional[NormalRole] = None
    cut_off_index = None # Index to truncate conversation history if ResumeFrom is encountered

    for i, session_event in enumerate(session.events): # Iterate through SessionEvents
        event = session_event.body # Access EventBody from SessionEvent
        if isinstance(event, ResumeFrom):
            cut_off_index = _find_event_index_by_id(session.events, event.from_event_id)
            if cut_off_index is not None:
                print(f"ResumeFrom found, cutting off history after event with id: {event.from_event_id} at index {cut_off_index}")
                # Truncate session events up to (and including) the cut-off event.
                session.events = session.events[:cut_off_index+1] # exclusive of the index
                break # Only handle the first ResumeFrom for now. In principle, there should only be one in a given session processing cycle.
            else:
                print(f"Warning: ResumeFrom refers to unknown event id: {event.from_event_id}. Ignoring.")
                continue # Skip to next event

    for session_event in session.events: # Iterate through SessionEvents again (possibly truncated)
        event = session_event.body # Access EventBody from SessionEvent
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

    # Validate all assistant messages
    for msg in conv:
        if msg["role"] == "assistant":
            validate_flattened_assistant_msg(msg)

    return conv


def _find_event_index_by_id(session_events: List[SessionEvent], event_id: str) -> Optional[int]: # Expects List[SessionEvent]
    """Helper function to find the index of an event by its ID."""
    for index, session_event in enumerate(session_events): # Iterate through SessionEvents
        if session_event.event_id == event_id: # Access event_id from SessionEvent
            return index
    return None # Not found
