"""
Code syntax highlighting, HTML output.
"""

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


def highlight_code(code: str) -> str:
    lexer = get_lexer_by_name("python")

    formatter = HtmlFormatter(
        style="monokai",  # Dark theme by default
        cssclass="highlight",
        wrapcode=True,
    )

    return highlight(code, lexer, formatter)


def get_pygments_css() -> str:
    """Get Pygments CSS for both light and dark themes."""
    # Base styles from HtmlFormatter
    formatter = HtmlFormatter()
    base_css = formatter.get_style_defs(".highlight")

    # Add media queries for light/dark theme support
    return f"""
    /* Light theme */
    @media (prefers-color-scheme: light) {{
        .highlight {{ background: #f8f8f8; }}
        {base_css}
    }}

    /* Dark theme */
    @media (prefers-color-scheme: dark) {{
        .highlight {{ background: #272822; }}
        {base_css}
    }}
    """
