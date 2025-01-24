from dataclasses import dataclass
from typing import List

from .event.types import EventBody


@dataclass
class Session:
    # For now, just body.
    # But going to be hard to have back-refs without IDs.
    events: List[EventBody]
