"""
Structured interpretation of LLM output.
"""

import traceback
from typing import List
import re
import markdown2
import bs4
from bs4 import BeautifulSoup
from torch._C import Event

from src.types import EventBody, AssistantThought, AssistantAction, CodeFragment

def extract_tag_content(text: str, tag: str) -> tuple[str, str]:
    """
    Extract content between XML-style tags and return (content, remaining_text).
    The remaining text has the full tag removed.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return "", text

    content = match.group(1).strip()
    remaining = text[:match.start()] + text[match.end():]
    return content, remaining

def extract_code_from_markdown(md_text: str) -> str:
    """Extract the one and only code block from markdown text."""
    # Use markdown2 to parse the text
    html = markdown2.markdown(md_text, extras=['fenced-code-blocks'])
    soup = BeautifulSoup(html, 'html.parser')

    # Find all code blocks
    code_blocks = soup.find_all('code')

    # Ensure there's exactly one
    if not code_blocks:
        raise ValueError("No code block found in markdown")
    if len(code_blocks) > 1:
        raise ValueError("Multiple code blocks found in markdown")

    # Return the content of the only code block
    return code_blocks[0].get_text()

def parse_constrained_message(text: str) -> List[EventBody]:
    """
    Parse a message from the constrained generation into a sequence of events.
    Expected format: <thought>...</thought> <action>...</action> ```python\n...\n```
    """
    try:
        # Extract thought, get remaining text
        thought_text, remaining = extract_tag_content(text, "thought")
        events: List[EventBody] = [AssistantThought(thought_text)]

        # Extract action, get remaining text
        action_text, remaining = extract_tag_content(remaining, "action")
        events.append(AssistantAction(action_text))

        # Clean up remaining text and parse as markdown
        remaining = remaining.strip()
        code = extract_code_from_markdown(remaining)
        events.append(CodeFragment(code))
    except Exception as e:
        print("Message parsing failed for message:", text)
        raise ValueError(f"Failed to parse constrained message: {e}")
        traceback.print_exc()

    return events
