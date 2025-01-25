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
# Autoreload local code.
# %load_ext autoreload
# %autoreload 2

# %%
import sys

# %%
# %pwd

# %%
sys.path.append('..')

# %%
from src.persist.load import session_from_file
from src.preproc import session_to_chatml

# %%
path = '../data/c0.xml'
session = session_from_file(path)
conversation = session_to_chatml(session)

# %%
session

# %%
conversation

# %%
print_conversation(conversation)
