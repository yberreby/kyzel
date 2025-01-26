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

import sys
sys.path.append('..')

# %% [markdown]
# ## Model Setup

# %%
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../run/ckpt/phi4_lora",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Setup tokenizer with phi-4 chat template
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

# %%
# What comes after is rather broken for now. Seem to be upstream issues.

# %%
# 16-bit merged saving
#model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)

# %%
!(cd llama.cpp; cmake -B build;cmake --build build --config Release && cp ./build/bin/* ./)

# %%
# Retry after build
model.save_pretrained_gguf("../run/gguf/phi4_lora_q8", tokenizer, quantization_method = "q8_0")

# %%
