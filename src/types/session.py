"""
Session = (roughly) an ordered collection of events + metadata.
"""

from dataclasses import dataclass
from typing import List

from .events import EventBody


@dataclass
class Session:
    # For now, just body.
    # But going to be hard to have back-refs without IDs.
    events: List[EventBody]
