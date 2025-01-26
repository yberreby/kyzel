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

# %% [markdown]
# # Kyzel Model Training

# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('..')

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

# %% [markdown]
# ## Model Setup

# %%
# Initialize model and tokenizer
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Setup tokenizer with phi-4 chat template
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

# %% [markdown]
# ## Dataset Loading and Analysis

# %%
# Load dataset
dataset = load_from_disk("../data/training_sessions_hf")
print(f"Loaded dataset with {len(dataset)} examples")

# Analyze token lengths
lengths = []
for sample in tqdm(dataset):
    tokens = tokenizer(sample["text"], return_tensors="pt", truncation=False)
    lengths.append(len(tokens.input_ids[0]))

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, edgecolor='black')
plt.title('Distribution of Sample Lengths (in tokens)')
plt.xlabel('Length in Tokens')
plt.ylabel('Count')

# Add stats
stats = f"""
Mean: {np.mean(lengths):.1f}
Median: {np.median(lengths):.1f}
Max: {np.max(lengths)}
95th %ile: {np.percentile(lengths, 95):.1f}
"""
plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Dataset Splitting

# %%
# Split if more than 1 sample
n_samples = len(dataset)
if n_samples > 1:
    eval_size = min(max(1, int(0.1 * n_samples)), n_samples - 1)
    splits = dataset.train_test_split(test_size=eval_size, seed=3407)
    train_dataset = splits['train']
    eval_dataset = splits['test']
    print(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
else:
    print("Single sample - using for training only")
    train_dataset = dataset
    eval_dataset = None

# %%
train_dataset['file']

# %%
eval_dataset['file']

# %% [markdown]
# ## Training

# %%
# Configure training
config_args = {
    "learning_rate": 2e-4,
    "weight_decay": 1, # yep.
    "warmup_steps": 5,
    "num_train_epochs": 20,
    "max_seq_length": max_seq_length,
    "dataset_num_proc": 1,
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "optim": "adamw_8bit",
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "../run/model_training_outputs",
    "report_to": "none",
    "packing": False,
}

# Add eval config if we have eval data
if eval_dataset is not None:
    config_args.update({
        "evaluation_strategy": "steps",
        "eval_steps": 1,
        "save_strategy": "steps",
        "save_steps": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
    })

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=SFTConfig(**config_args),
)

# Only train on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

# %% [markdown]
# ## Execute Training

# %%
# Track GPU memory
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}")
print(f"Initial reserved memory: {start_gpu_memory} GB")

# Train
trainer_stats = trainer.train()

# Print stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f} seconds")
print(f"Peak memory usage: {used_memory} GB (LoRA: {used_memory_for_lora} GB)")

# %%
model.save_pretrained("../run/ckpt/phi4_lora")
tokenizer.save_pretrained("../run/ckpt/phi4_lora")

# %%
