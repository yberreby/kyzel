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
    individual_event_types,
)
from src.run.format import LLMFormatter
from .css import get_base_css
from .md import format_markdown
from .highlight import highlight_code, get_pygments_css


def get_full_css() -> str:
    """Combine base CSS with Pygments syntax highlighting CSS."""
    return f"{get_base_css()}\n{get_pygments_css()}"


def event_html_inner(session_event: SessionEvent) -> str:
    """Generate HTML for a single event without CSS."""
    event = session_event.body
    event_id_str = session_event.event_id
    event_id_html = (
        f"<span class='event-id'>Event ID: {event_id_str}</span>"
        if event_id_str
        else ""
    )

    match event:
        case HumanMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event human'>{formatted}{event_id_html}</div>"
        case AssistantThought():
            formatted = format_markdown(event.text)
            return (
                f"<div class='event assistant-thought'>{formatted}{event_id_html}</div>"
            )
        case CodeFragment():
            highlighted = highlight_code(event.code)
            return f"<div class='event code'>{highlighted}{event_id_html}</div>"
        case AssistantMsg():
            formatted = format_markdown(event.text)
            return f"<div class='event assistant'>{formatted}{event_id_html}</div>"
        case AssistantAction():
            return f"<div class='event action'>{event.text}{event_id_html}</div>"
        case ExecutionResult():
            if event.success:  # Access success directly from ExecutionResult
                formatted_result = LLMFormatter.format_result(event)  # Use LLMFormatter
                output_content = formatted_result.output  # Access formatted output
                output_html = f"<div class='execution-output'>{output_content}</div>"  # Display formatted output
            else:
                error_html = f"<div class='execution-error'><b>Error:</b><pre>{event.error_traceback}</pre></div>"
                output_html = error_html

            return f"<div class='event execution'>{output_html}{event_id_html}</div>"
        case _:
            warn(
                f"Unknown event type: {type(event)}, falling back to `repr` for display"
            )
            return repr(event)


def format_event(session_event: SessionEvent) -> str:
    """Format a single event with CSS."""
    return f"<style>{get_full_css()}</style>{event_html_inner(session_event)}"


def format_session(session: Session) -> str:
    """Format an entire session with CSS."""
    events_html = "\n".join(event_html_inner(e) for e in session.events)
    return f'<style>{get_full_css()}</style><div class="session">{events_html}</div>'


def register_formatters() -> None:
    """Register HTML formatters for types with IPython."""
    ip = get_ipython()
    if ip is None:
        return

    html_formatter = ip.display_formatter.formatters["text/html"]

    html_formatter.for_type(Session, format_session)

    for event_type in individual_event_types:
        html_formatter.for_type(SessionEvent, format_event)
