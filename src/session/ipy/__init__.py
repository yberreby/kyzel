"""
Nice session display in Jupyter.
"""

from warnings import warn

from IPython import get_ipython

from ..types import Session
from ..event.types import (
    AssistantAction,
    HumanMsg,
    AssistantThought,
    CodeFragment,
    AssistantMsg,
    ExecutionResult,
    EventBody,
)
from ..event import individual_event_types

from .css import get_base_css
from .md import format_markdown
from .highlight import highlight_code, get_pygments_css


def get_full_css() -> str:
    """Combine base CSS with Pygments syntax highlighting CSS."""
    return f"{get_base_css()}\n{get_pygments_css()}"


def event_html_inner(event: EventBody) -> str:
    """Generate HTML for a single event without CSS."""
    match event:
        case HumanMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event human'>{formatted}</div>"
        case AssistantThought():
            formatted = format_markdown(event.text)
            return f"<div class='event assistant-thought'>{formatted}</div>"
        case CodeFragment():
            highlighted = highlight_code(event.code)
            return f"<div class='event code'>{highlighted}</div>"
        case AssistantMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event assistant'>{formatted}</div>"
        case AssistantAction():
            # Just a title, no need for markdown
            return f"<div class='event action'>{event.text}</div>"
        case ExecutionResult():
            # Keep execution results verbatim
            return f"<div class='event execution'>{event.output}</div>"
        case _:
            warn(
                f"Unknown event type: {type(event)}, falling back to `repr` for display"
            )
            return repr(event)


def format_event(event: EventBody) -> str:
    """Format a single event with CSS included."""
    return f"<style>{get_full_css()}</style>{event_html_inner(event)}"


def format_session(session: Session) -> str:
    """Format an entire session with CSS included."""
    events_html = "\n".join(event_html_inner(e) for e in session.events)
    return f'<style>{get_full_css()}</style><div class="session">{events_html}</div>'


def register_formatters() -> None:
    """Register HTML formatters for our types with IPython."""
    ip = get_ipython()
    if ip is None:
        return

    html_formatter = ip.display_formatter.formatters["text/html"]

    # Register Session formatter
    html_formatter.for_type(Session, format_session)

    # Register individual event formatters.
    # We can't just do it on the union type EventBody - it doesn't have __name__.
    for event_type in individual_event_types:
        html_formatter.for_type(event_type, format_event)
