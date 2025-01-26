"""
Session serialization to XML.
"""
from src.types import Session, SessionEvent, EventBody, HumanMsg, AssistantMsg, AssistantThought, AssistantAction, CodeFragment, ExecutionResult
import xml.etree.ElementTree as ET
from xml.dom import minidom  # For pretty printing XML


def event_to_xml(session_event: SessionEvent) -> ET.Element:
    """
    Converts a SessionEvent to its XML representation.
    """
    event = session_event.body
    attrib = {}
    if session_event.event_id:
        attrib['id'] = session_event.event_id

    if isinstance(event, HumanMsg):
        el = ET.Element("msg", attrib={"from": "user", **attrib})
        el.text = event.text
    elif isinstance(event, AssistantMsg):
        el = ET.Element("msg", attrib={"from": "assistant", **attrib})
        el.text = event.text
    elif isinstance(event, AssistantThought):
        el = ET.Element("thought", attrib=attrib)
        el.text = event.text
    elif isinstance(event, AssistantAction):
        el = ET.Element("action", attrib=attrib)
        el.text = event.text
    elif isinstance(event, CodeFragment):
        el = ET.Element("code", attrib=attrib) # Correct tag for CodeFragment
        el.text = event.code
    elif isinstance(event, ExecutionResult):
        el = ET.Element("result", attrib=attrib) # Correct tag for ExecutionResult
        el.text = event.output
    elif isinstance(event, ResumeFrom):
        el = ET.Element("resume_from", attrib=attrib) # New tag for ResumeFrom
        el.set('from_event_id', event.from_event_id) # Serialize from_event_id as attribute
    else:
        raise ValueError(f"Unknown event type: {type(event)}")
    return el


def to_xml_str(session: Session, pretty_print=True) -> str:
    """
    Serializes a Session object to an XML string representation.
    """
    session_el = ET.Element("session")
    events_el = ET.SubElement(session_el, "events")

    for session_event in session.events:
        event_el = event_to_xml(session_event)
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
