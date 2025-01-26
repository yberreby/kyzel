"""
Clean and carefully written XML -> Python ingestion routines.

For individual events.
"""

from xml.etree.ElementTree import Element as XmlElement
from src.types import (
    HumanMsg,
    AssistantMsg,
    AssistantThought,
    AssistantAction,
    CodeFragment,
    ExecutionResult,
    EventBody,
    ResumeFrom, # Import ResumeFrom
    SessionEvent, # Import SessionEvent
)
import uuid # for generating event_ids if missing in XML


def normalized_text(el: XmlElement) -> str:
    # Removing leading and trailing newlines.
    # EXCEPT for code output, they are essentially devoid of meaning.
    # With low-data training, this is an unnecessary potential source of inconsistency.
    if not el.text:
        return "" # Allow empty text for robustness (e.g., empty <result></result>)

    return el.text.strip()


def msg_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    assert el.tag == "msg"
    sender = el.get("from")
    content = normalized_text(el)
    event_id_str = el.get("id") # load event_id

    body: EventBody
    match sender:
        case "user":
            body = HumanMsg(text=content)
        case "assistant":
            body = AssistantMsg(text=content)
        case _:
            raise ValueError(f"unknown sender {sender} for <msg>")

    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body) # Wrap in SessionEvent


def thought_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    assert el.tag == "thought"
    text = normalized_text(el)
    event_id_str = el.get("id") # load event_id
    body = AssistantThought(text=text)
    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body) # Wrap in SessionEvent


def code_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    assert el.tag == "code"
    python_src = el.text # No normalization for code
    event_id_str = el.get("id") # load event_id
    body = CodeFragment(code=python_src)
    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body) # Wrap in SessionEvent


def exec_result_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    assert el.tag == "result"
    output = el.text if el.text is not None else "" # No normalization for result
    event_id_str = el.get("id") # load event_id
    body = ExecutionResult(output=output)
    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body) # Wrap in SessionEvent


def action_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    assert el.tag == "action"
    text = normalized_text(el)
    event_id_str = el.get("id") # load event_id
    body = AssistantAction(text=text)
    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body) # Wrap in SessionEvent


def resume_from_event_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent
    assert el.tag == "resume_from"
    from_event_id = el.get("from_event_id")
    if not from_event_id:
        raise ValueError("<resume_from> must have 'from_event_id' attribute")
    event_id_str = el.get("id") # load event_id
    body = ResumeFrom(from_event_id=from_event_id)
    event_id = event_id_str or str(uuid.uuid4()) # Generate if missing
    return SessionEvent(event_id=event_id, body=body)


def event_from_xml(el: XmlElement) -> SessionEvent: # Returns SessionEvent now
    """
    Convert the parsed XML representing a single event, such as '<msg>...</msg>', into the corresponding Python type.
    """
    match el.tag:
        case "msg":
            return msg_from_xml(el)
        case "thought":
            return thought_from_xml(el)
        case "code":
            return code_from_xml(el)
        case "result":
            return exec_result_from_xml(el)
        case "action":
            return action_from_xml(el)
        case "resume_from":
            return resume_from_event_from_xml(el) # Handle ResumeFrom
        case _:
            raise ValueError(f"unknown event XML tag: '{el.tag}")
