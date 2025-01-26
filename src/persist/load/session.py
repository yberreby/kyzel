"""
Parsing/loading an entire session.
"""

from typing import List
import xml.etree.ElementTree as ET
from src.types import Session, SessionEvent # Import SessionEvent
from .event import event_from_xml


def from_str(xml_str: str) -> Session:
    session_el = ET.fromstring(xml_str)
    xml_events = session_el.find(".//events")

    if xml_events is None:
        raise ValueError("Missing <events>")

    events: List[SessionEvent] = []
    for xml_event in xml_events:
        session_event = event_from_xml(xml_event) # Now returns SessionEvent
        events.append(session_event)
    return Session(events=events)


def from_file(path) -> Session:
    with open(path, "r") as f:
        xml_str = f.read()
    return from_str(xml_str)
