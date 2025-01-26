"""XML -> Python ingestion routines for individual events."""

from xml.etree.ElementTree import Element as XmlElement
from src.types import (
    HumanMsg,
    AssistantMsg,
    AssistantThought,
    AssistantAction,
    CodeFragment,
    ExecutionResult,
    EventBody,
    ResumeFrom,
    SessionEvent,
)
import uuid
from src.run.execute import (
    CellOutput,
)  # Import CellOutput for consistency with unified ExecutionResult


def normalized_text(el: XmlElement) -> str:
    """Normalize text content from XML element."""
    if not el.text:
        return ""
    return el.text.strip()


def msg_from_xml(el: XmlElement) -> SessionEvent:
    """Load msg event from XML."""
    assert el.tag == "msg"
    sender = el.get("from")
    content = normalized_text(el)
    event_id_str = el.get("id")

    body: EventBody
    match sender:
        case "user":
            body = HumanMsg(text=content)
        case "assistant":
            body = AssistantMsg(text=content)
        case _:
            raise ValueError(f"unknown sender {sender} for <msg>")

    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def thought_from_xml(el: XmlElement) -> SessionEvent:
    """Load thought event from XML."""
    assert el.tag == "thought"
    text = normalized_text(el)
    event_id_str = el.get("id")
    body = AssistantThought(text=text)
    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def code_from_xml(el: XmlElement) -> SessionEvent:
    """Load code event from XML."""
    assert el.tag == "code"
    python_src = el.text
    event_id_str = el.get("id")
    body = CodeFragment(code=python_src)
    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def exec_result_from_xml(el: XmlElement) -> SessionEvent:
    """Load execution result event from XML."""
    assert el.tag == "result"
    # For simplicity, when loading from XML, we will only store the output string in CellOutput.stdout
    output_str = el.text if el.text is not None else ""
    cell_output = CellOutput(
        stdout=output_str, stderr="", display_output="", result=None
    )  # Create CellOutput
    body = ExecutionResult(
        output=cell_output, success=True
    )  # Use unified ExecutionResult, default to success=True when loading from XML
    event_id_str = el.get("id")
    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def action_from_xml(el: XmlElement) -> SessionEvent:
    """Load action event from XML."""
    assert el.tag == "action"
    text = normalized_text(el)
    event_id_str = el.get("id")
    body = AssistantAction(text=text)
    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def resume_from_event_from_xml(el: XmlElement) -> SessionEvent:
    """Load resume_from event from XML."""
    assert el.tag == "resume_from"
    from_event_id = el.get("from_event_id")
    if not from_event_id:
        raise ValueError("<resume_from> must have 'from_event_id' attribute")
    event_id_str = el.get("id")
    body = ResumeFrom(from_event_id=from_event_id)
    event_id = event_id_str or str(uuid.uuid4())
    return SessionEvent(event_id=event_id, body=body)


def event_from_xml(el: XmlElement) -> SessionEvent:
    """Convert XML event element to SessionEvent."""
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
            return resume_from_event_from_xml(el)
        case _:
            raise ValueError(f"unknown event XML tag: '{el.tag}")
