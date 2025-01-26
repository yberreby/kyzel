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
from src.types import Session, HumanMsg, ExecutionResult, CodeFragment, ResumeFrom, SessionEvent, AssistantThought, AssistantAction
import uuid # Import uuid for generating event IDs


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


def _process_session_events():
    global session

    # Convert session to ChatML
    human_and_assistant_msgs = session_to_chatml(session)

    # Apply system message
    system_msg = {
        "role": "system",
        "content": "You are an IPython REPL assistant. Think to yourself in <thought>, succinctly (in 1-5 words) state what your next code block will do in <action>, then output a Python code block, whose results will be returned to you. Imports, variables, etc, are persistent. In your code, do not use comments. Reuse previously-defined variables, previous imports, etc. Avoid defining functions. Write as you would in Jupyter notebook, but use display() or print() explicitly."
    }
    conversation = [system_msg] + human_and_assistant_msgs

    raw_response = llm.generate(conversation, max_new_tokens=max_new_tokens)
    # Uncomment for debugging:
    # print("Raw LLM response:", raw_response)

    # Parse response into events
    assistant_events_body = parse_constrained_message(raw_response)
    assistant_session_events = [SessionEvent(event_id=str(uuid.uuid4()), body=e) for e in assistant_events_body]

    # Add all events to session
    session.events.extend(assistant_session_events)

    # Clear previous output and display current state
    clear_output(wait=True)
    display(session)

    # Find code fragment to execute (if any)
    code_fragments = [se.body for se in assistant_session_events if isinstance(se.body, CodeFragment)]
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
    result_event_body = ExecutionResult(formatter.format_result(result).output)
    result_session_event = SessionEvent(event_id=str(uuid.uuid4()), body=result_event_body)
    session.events.append(result_session_event)

    # Final display
    #clear_output(wait=True) # Let's keep the output for now for better flow.
    display(session)



def process_user_input(query: str):
    """Process a single user query and update session."""
    global session

    # Create and add user message, with a new event ID
    msg = HumanMsg(query)
    session_event = SessionEvent(event_id=str(uuid.uuid4()), body=msg) # Generate event ID
    session.events.append(session_event)
    _process_session_events()




def regenerate_assistant_response():
    """Regenerate the assistant's last response."""
    global session

    # Remove assistant events since the last user message.
    # Go backwards
    original_len = len(session.events)
    i = len(session.events) - 1
    regenerate_from_event_id = None
    while i >= 0:
        session_event = session.events[i]
        print(session_event)
        match session_event.body:
            case AssistantThought() | AssistantAction() | CodeFragment() | ExecutionResult():
                session.events.pop(i) # pop in place
            case _:
                regenerate_from_event_id = session_event.event_id # Get event ID from SessionEvent
                break
        i -= 1
    removed_count = original_len - len(session.events)
    print(f"Removed {removed_count} assistant events for regeneration.")
    input()

    # Add a ResumeFrom.
    if regenerate_from_event_id: # should always be the case, but for robustness.
        resume_event = ResumeFrom(from_event_id=regenerate_from_event_id)
        session_event = SessionEvent(event_id=str(uuid.uuid4()), body=resume_event) # Generate event ID for resume event
        session.events.append(session_event)

    # Re-run the processing with the current session history.
    if session.events:
        _process_session_events() # re-process from the current history
    else:
        print("No user query found in session to regenerate response for.")


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
            print("- 'execute' or 'x': Execute code from the last assistant response (if any)") # More discoverable
            print("- Ctrl+C: Exit interactive mode")
            print("\nFor other input, enter your query directly.")
        elif query.strip().lower() in ['regenerate', 'r']:
            regenerate_assistant_response()
        elif query.strip().lower() in ['save', 's']:
            save_session()
        elif query.strip():  # avoid processing empty queries
            process_user_input(query) # initial query
        elif query.strip().lower() in ['execute', 'x']: # Making execution more discoverable
            # Find code fragment (if any) to execute from the *last* assistant response
            assistant_events = [se for se in reversed(session.events) if event_source_role(se.body) == "assistant"] # Check body role
            if not assistant_events:
                print("No assistant messages found in history.")
                continue
            last_assistant_msg = assistant_events[0] # should be the most recent
            code_fragments = [se.body for se in assistant_events if isinstance(se.body, CodeFragment)] # Extract CodeFragment bodies from SessionEvents
            if not code_fragments:
                print("No code to execute in the last assistant response.")
                continue

            # Ask for execution permission
            response = input("\nExecute code from last assistant response? [y/N] ").strip().lower()
            if response != 'y':
                print("Execution skipped")
                continue

            # Execute the code (we can reuse the execution logic from process_user_input)
            code_to_execute = code_fragments[0].code # just take the first one for now
            result = executor.execute(code_to_execute)
            result_event_body = ExecutionResult(formatter.format_result(result).output)
            result_session_event = SessionEvent(event_id=str(uuid.uuid4()), body=result_event_body) # Generate event ID for result
            session.events.append(result_session_event)
    except KeyboardInterrupt:
            print("\nExiting...")
            break

# %%
# %pwd

# %%
import requests

# %%
