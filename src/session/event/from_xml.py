"""
Clean and carefully written XML -> Python ingestion routines.
"""

from xml.etree.ElementTree import Element as XmlElement
from .types import (
    HumanMsg,
    AssistantMsg,
    AssistantThought,
    CodeFragment,
    ExecutionResult,
    EventBody,
)


__all__ = [
    "HumanMsg",
    "AssistantThought",
    "CodeFragment",
    "AssistantMsg",
    "ExecutionResult",
    "EventBody",
]


def normalized_text(el: XmlElement) -> str:
    # Removing leading and trailing newlines.
    # EXCEPT for code output, they are essentially devoid of meaning.
    # With low-data training, this is an unnecessary potential source of inconsistency.
    if not el.text:
        raise ValueError(f"empty .text in {el.tag} element")

    return el.text.strip()


def msg_from_xml(el: XmlElement) -> HumanMsg | AssistantMsg:
    assert el.tag == "msg"
    sender = el.get("from")
    content = normalized_text(el)

    match sender:
        case "user":
            return HumanMsg(content)
        case "assistant":
            return AssistantMsg(content)
        case _:
            raise ValueError(f"unknown sender {sender} for <msg>")


def thought_from_xml(el: XmlElement) -> AssistantThought:
    assert el.tag == "thought"
    return AssistantThought(normalized_text(el))


def code_from_xml(el: XmlElement) -> CodeFragment:
    assert el.tag == "code"

    # Normalizing here is debatable, but for now, let's do it.
    python_src = normalized_text(el)
    return CodeFragment(python_src)


def exec_result_from_xml(el: XmlElement) -> ExecutionResult:
    assert el.tag == "result"

    # For now, result is just a string.
    # Could be enriched, dissociate stdout and stderr...

    # Very debatable normalization, but again, for now, pros outweigh cons.
    output = normalized_text(el) if el.text else ""

    return ExecutionResult(output)


def event_from_xml(el: XmlElement) -> EventBody:
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
        case _:
            raise ValueError(f"unknown event XML tag: '{el.tag}")
