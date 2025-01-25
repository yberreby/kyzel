"""
ChatML-specific types.
"""

from typing import Literal, TypedDict

# Restrictive type. Most of the time, no system messages in sight.
NormalRole = Literal["user", "assistant"]

Role = Literal["system"] | NormalRole


class Msg(TypedDict):
    """
    A single ChatML message.
    """

    role: Role
    content: str


Conversation = list[Msg]
