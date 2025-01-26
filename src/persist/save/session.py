"""
Session serialization to XML.
"""
from src.types import Session, EventBody, HumanMsg, AssistantMsg, AssistantThought, AssistantAction, CodeFragment, ExecutionResult
import xml.etree.ElementTree as ET
from xml.dom import minidom  # For pretty printing XML


def event_to_xml(event: EventBody) -> ET.Element:
    """
    Converts an EventBody to its XML representation.
    """
    if isinstance(event, HumanMsg):
        el = ET.Element("msg", attrib={"from": "user"})
        el.text = event.text
    elif isinstance(event, AssistantMsg):
        el = ET.Element("msg", attrib={"from": "assistant"})
        el.text = event.text
    elif isinstance(event, AssistantThought):
        el = ET.Element("thought")
        el.text = event.text
    elif isinstance(event, AssistantAction):
        el = ET.Element("action")
        el.text = event.text
    elif isinstance(event, CodeFragment):
        el = ET.Element("code")
        el.text = event.code
    elif isinstance(event, ExecutionResult):
        el = ET.Element("result")
        el.text = event.output
    else:
        raise ValueError(f"Unknown event type: {type(event)}")
    return el


def to_xml_str(session: Session, pretty_print=True) -> str:
    """
    Serializes a Session object to an XML string representation.
    """
    session_el = ET.Element("session")
    events_el = ET.SubElement(session_el, "events")

    for event in session.events:
        event_el = event_to_xml(event)
        events_el.append(event_el)

    if pretty_print:
        rough_string = ET.tostring(session_el, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    else:
        return ET.tostring(session_el, encoding='unicode')


def to_file(session: Session, path: str, pretty_print=True):
    """
    Serializes a Session object to an XML file.
    """
    xml_str = to_xml_str(session, pretty_print=pretty_print)
    with open(path, "w") as f:
        f.write(xml_str)
