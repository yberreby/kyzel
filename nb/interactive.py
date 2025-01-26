# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append('..')

# %%
from src.generate.llm import LLM
from src.persist.load import session_from_file
from src.persist.save import to_file
from src.preproc import session_to_chatml, event_source_role
from src.postproc import parse_constrained_message
from src.run.execute import IPythonExecutor
from src.run.format import LLMFormatter
from src.types import Session, HumanMsg, ExecutionResult, CodeFragment

# %%
max_seq_length = 16000
max_new_tokens = 2048

# %%
llm = LLM(max_seq_length=max_seq_length)

# %%
from IPython.display import clear_output

executor = IPythonExecutor()
formatter = LLMFormatter()
session = Session(events=[])


def process_user_input(query: str):
    """Process a single user query and update session."""
    global session

    # Create and add user message
    msg = HumanMsg(query)
    session.events.append(msg)

    # Convert session to ChatML
    human_and_assistant_msgs = session_to_chatml(session)

    # Apply system message
    system_msg = {
        "role": "system",
        "content": "You are an IPython REPL assistant. Think to yourself in <thought>, succinctly (in 1-5 words) state what your next code block will do in <action>, then output a Python code block, whose results will be returned to you. Imports, variables, etc, are persistent. In your code, do not use comments. Reuse previously-defined variables, previous imports, etc. Avoid defining functions. Write as you would in Jupyter notebook."
    }
    conversation = [system_msg] + human_and_assistant_msgs

    raw_response = llm.generate(conversation, max_new_tokens=max_new_tokens)
    # Uncomment for debugging:
    # print("Raw LLM response:", raw_response)

    # Parse response into events
    events = parse_constrained_message(raw_response)  # potentially raises ValueError

    # Add all events to session
    session.events.extend(events)

    # Clear previous output and display current state
    clear_output(wait=True)
    display(session)

    # Find code fragment (if any) to execute
    code_fragments = [e for e in events if isinstance(e, CodeFragment)]
    if not code_fragments:
        print("No code to execute!")
        return

    # Ask for execution permission
    response = input("\nExecute? [y/N] ").strip().lower()
    if response != 'y':
        print("Execution skipped")
        return

    # Execute and add result
    result = executor.execute(code_fragments[0].code)
    result_event = ExecutionResult(formatter.format_result(result).output)
    session.events.append(result_event)

    # Final display
    clear_output(wait=True) # Let's keep the output for now for better flow.
    display(session)


def regenerate_assistant_response():
    """Regenerate the assistant's last response."""
    global session

    # Remove assistant events since the last human message.
    # This assumes that the session ends with a HumanMsg and then Assistant's response.
    while session.events and event_source_role(session.events[-1]) == "assistant":
        session.events.pop()

    # Re-run the processing with the current session history.
    if session.events:
        last_human_msg = [e for e in reversed(session.events) if isinstance(e, HumanMsg)][
            0]  # should always find one
        process_user_input(last_human_msg.text)  # re-process the last query
    else:
        print("No user query found to regenerate response for.")


def save_session():
    """Save the current session to a file."""
    import datetime
    import os

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.xml"
    save_dir = "../data/sessions"  # save to data/sessions by default
    os.makedirs(save_dir, exist_ok=True)  # ensure directory exists
    save_path = os.path.join(save_dir, filename)

    to_file(session, save_path)
    print(f"Session saved to: {save_path}")


# Interactive loop
print("Interactive IPython REPL assistant. Type 'help' for commands.")
while True:
    try:
        query = input("\nEnter query ('help' for commands, Ctrl+C to exit): ")
        if query.strip().lower() == 'help':
            print("\nAvailable commands:")
            print("- 'help': Show this help message")
            print("- 'regenerate' or 'r': Regenerate assistant's last response")
            print("- 'save' or 's': Save the current session")
            print("- Ctrl+C: Exit interactive mode")
            print("\nFor other input, enter your query directly.")
        elif query.strip().lower() in ['regenerate', 'r']:
            regenerate_assistant_response()
        elif query.strip().lower() in ['save', 's']:
            save_session()
        elif query.strip():  # avoid processing empty queries
            process_user_input(query)
    except KeyboardInterrupt:
            print("\nExiting...")
            break

# %%
