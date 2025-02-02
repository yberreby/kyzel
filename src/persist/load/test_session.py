from .session import from_str
from src.types import (
    HumanMsg,
    AssistantThought,
    CodeFragment,
    AssistantMsg,
    ExecutionResult,
)


def test_parse():
    xml = """<session>
        <events>
            <msg from="user">Help me parse this CSV file</msg>
            <thought>I'll use pandas to read the file efficiently</thought>
            <code>import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())</code>
            <result>   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9</result>
            <msg from="assistant">I've loaded your CSV. The first three columns are named A, B, and C. What would you like to do with this data?</msg>
        </events>
    </session>
    """

    # Should not crash.
    session = from_str(xml)
    events = session.events

    # Correct event number and types.
    assert len(events) == 5
    assert isinstance(events[0].body, HumanMsg)
    assert isinstance(events[1].body, AssistantThought)
    assert isinstance(events[2].body, CodeFragment)
    assert isinstance(events[3].body, ExecutionResult)
    assert isinstance(events[4].body, AssistantMsg)

    # No leading newline.
    assert events[2].body.code.startswith("import pandas")
    assert "df.head()" in events[2].body.code
    # Inner whitespace preserved.
    assert "A  B  C" in events[3].body.output.stdout
