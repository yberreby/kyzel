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
from src.session import from_file, flatten_to_chatml
from src.chatml import print_conversation

# %%
path = '../data/c0.xml'
session = from_file(path)

# %%
session

# %%
conversation = flatten_to_chatml(session)

# %%
conversation

# %%
print_conversation(conversation)
