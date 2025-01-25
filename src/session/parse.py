import xml.etree.ElementTree as ET

from .event import event_from_xml
from .types import Session


def from_str(xml_str: str) -> Session:
    session_el = ET.fromstring(xml_str)
    xml_events = session_el.find(".//events")

    if xml_events is None:
        raise ValueError("Missing <events>")

    events = [event_from_xml(e) for e in xml_events]
    return Session(events=events)


def from_file(path) -> Session:
    with open(path, "r") as f:
        xml_str = f.read()
    return from_str(xml_str)
