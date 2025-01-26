"""
Session = (roughly) an ordered collection of events + metadata.
"""

from dataclasses import dataclass
from typing import List, Optional

from .events import EventBody


@dataclass
class SessionEvent:
    event_id: Optional[str]
    body: EventBody


@dataclass
class Session:
    # For now, just body.
    # But going to be hard to have back-refs without IDs.
    events: List[SessionEvent]
