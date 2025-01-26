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
