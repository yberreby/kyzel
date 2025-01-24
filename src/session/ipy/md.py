import markdown2


def format_markdown(text: str) -> str:
    """Convert markdown to HTML with additional features."""
    extras = ["fenced-code-blocks", "tables", "break-on-newline", "header-ids"]
    return markdown2.markdown(text, extras=extras)
