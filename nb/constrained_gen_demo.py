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
# Avoid rerunning this, takes time.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, "phi-4")
FastLanguageModel.for_inference(model);

# %%
from src.generate.constrain import StructuredEnforcer
def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")
    
    print("Input template:\n" + tokenizer.decode(inputs[0]))
    
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        logits_processor=[StructuredEnforcer(tokenizer)],
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# %%
# Test
response = generate("Output some Python code with triple backticks INSIDE of it (printing them for example). before that, just write <action>ACTING</action>")
print("\nFINAL OUTPUT:")
print(response)

# %%
from src.postproc import parse_constrained_message
parse_constrained_message(response)

# %%
for e in parse_constrained_message(response):
    display(e)

# %%
# This one is particularly evil.
response = generate("Output some Python code defining a triple-backtick template for string formatting, using a multiline triple-double-quote string. INDENT with leading whitespace the contents of that string containing the triple backquotes (markdown code fence).")
print("\nFINAL OUTPUT:")
print(response)

# %%
