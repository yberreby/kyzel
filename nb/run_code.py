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
from src.run.execute import IPythonExecutor
from src.run.format import LLMFormatter

# %%
# Core execution
executor = IPythonExecutor()
raw_result = executor.execute("raise ValueError('custom error')")

# %%
# Access rich error info
print(type(raw_result.error))  # <class 'ValueError'>
print(raw_result.error_traceback)  # Full traceback

# %%
# Format for LLM if needed
formatter = LLMFormatter()
llm_result = formatter.format_result(raw_result)
print(llm_result)  # Clean, formatted output

# %%
print(llm_result.error)  # Clean, formatted output

# %%
formatter.format_result(executor.execute("1+1"))

# %%
# Naked IPython experiments follow

# %%
from IPython import get_ipython
ipython = get_ipython()

# %%
result = ipython.run_cell("1+1")
print(f"result.success: {result.success}")
print(f"result.result: {result.result}")
print(f"result.error_in_exec: {result.error_in_exec}")
print(f"result.error_before_exec: {result.error_before_exec}")

# %%
result = ipython.run_cell("x = 1+1")
print(f"result.success: {result.success}")
print(f"result.result: {result.result}")
print(f"ipython.user_ns['x']: {ipython.user_ns.get('x')}")

# %%
result = ipython.run_cell("1+1")
print(f"result.success: {result.success}")
print(f"result.result: {result.result}")
print(f"ipython.last_execution_result.result: {ipython.last_execution_result.result}")

# %%
from IPython.utils.capture import capture_output
with capture_output(display=True) as captured:
    ipython.run_cell("1+1")

print(f"captured.stdout: {captured.stdout!r}")
print(f"captured.stderr: {captured.stderr!r}")
print(f"captured._outputs: {captured._outputs}")

# %%
