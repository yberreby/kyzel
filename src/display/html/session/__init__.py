"""
Nice session display in Jupyter.
"""

from warnings import warn

from IPython import get_ipython

from src.types import (
    Session,
    SessionEvent,
    AssistantAction,
    HumanMsg,
    AssistantThought,
    CodeFragment,
    AssistantMsg,
    ExecutionResult,
    EventBody,
    individual_event_types
)

from .css import get_base_css
from .md import format_markdown
from .highlight import highlight_code, get_pygments_css


def get_full_css() -> str:
    """Combine base CSS with Pygments syntax highlighting CSS."""
    return f"{get_base_css()}\n{get_pygments_css()}"


def event_html_inner(session_event: SessionEvent) -> str: # Expects SessionEvent now
    """Generate HTML for a single event without CSS."""
    event = session_event.body # Access EventBody from SessionEvent
    event_id_str = session_event.event_id # Access event_id from SessionEvent
    event_id_html = f"<span class='event-id'>Event ID: {event_id_str}</span>" if event_id_str else "" # Display event ID

    match event:
        case HumanMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event human'>{formatted}{event_id_html}</div>" # Include event ID
        case AssistantThought():
            formatted = format_markdown(event.text)
            return f"<div class='event assistant-thought'>{formatted}{event_id_html}</div>" # Include event ID
        case CodeFragment():
            highlighted = highlight_code(event.code)
            return f"<div class='event code'>{highlighted}{event_id_html}</div>" # Include event ID
        case AssistantMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event assistant'>{formatted}{event_id_html}</div>" # Include event ID
        case AssistantAction():
            # Just a title, no need for markdown
            return f"<div class='event action'>{event.text}{event_id_html}</div>" # Include event ID
        case ExecutionResult():
            # Keep execution results verbatim
            return f"<div class='event execution'>{event.output}{event_id_html}</div>" # Include event ID
        case _:
            warn(
                f"Unknown event type: {type(event)}, falling back to `repr` for display"
            )
            return repr(event)


def format_event(session_event: SessionEvent) -> str: # Expects SessionEvent now
    """Format a single event with CSS included."""
    return f"<style>{get_full_css()}</style>{event_html_inner(session_event)}" # Pass SessionEvent


def format_session(session: Session) -> str:
    """Format an entire session with CSS included."""
    events_html = "\n".join(event_html_inner(e) for e in session.events) # Iterate through SessionEvents
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
        # Need to format SessionEvent, not EventBody directly
        html_formatter.for_type(SessionEvent, format_event) # Register formatter for SessionEvent, not EventBody
