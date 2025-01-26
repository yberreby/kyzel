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
from src.persist.load.session import from_file

# %%
path = '../data/sessions/c0.xml'
display(from_file(path))

# %%
