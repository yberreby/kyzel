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
from src.preproc import session_to_chatml
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

    # Apply system message.
    system_msg = {
        "role": "system",
        "content": "You are an IPython REPL assistant. Think to yourself in <thought>, succinctly (in 1-5 words) state what your next code block will do in <action>, then output a Python code block, whose results will be returned to you. Imports, variables, etc, are persistent."
    }
    conversation = [system_msg] + human_and_assistant_msgs
    
    raw_response = llm.generate(conversation, max_new_tokens=max_new_tokens)
    # Uncomment for debugging
    #print("Raw:", raw_response)
    
    # Parse response into events
    events = parse_constrained_message(raw_response)
    
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
        
    # Ask for permission
    response = input("\nExecute? [y/N] ").strip().lower()
    if response != 'y':
        print("Execution skipped")
        return
        
    # Execute and add result
    result = executor.execute(code_fragments[0].code)
    result_event = ExecutionResult(formatter.format_result(result).output)
    session.events.append(result_event)
    
    # Final display
    #clear_output(wait=True)
    display(session)

# Interactive loop
while True:
    try:
        query = input("\nEnter query (or Ctrl+C to exit): ")
        process_user_input(query)
    except KeyboardInterrupt:
        print("\nExiting...")
        break

# %%
