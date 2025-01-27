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
# # Fine-tuning on session data

# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('..')

from warnings import warn
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
# ## Setup

# %%
import torch
from warnings import warn

# Check VRAM availability
if not torch.cuda.is_available():
    warn("No CUDA device detected. This code requires a GPU to run.")
    raise RuntimeError("GPU required")

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

# Change this to get good results at the price of speed.
# Set to None to auto-select based on VRAM
JUST_TESTING_USE_TERRIBLE_YET_FAST_MODEL = None

if JUST_TESTING_USE_TERRIBLE_YET_FAST_MODEL is None:
    # Auto-select based on VRAM
    JUST_TESTING_USE_TERRIBLE_YET_FAST_MODEL = vram_gb < 16
    if JUST_TESTING_USE_TERRIBLE_YET_FAST_MODEL:
        warn(f"Only {vram_gb:.1f}GB VRAM available - falling back to small model")

if JUST_TESTING_USE_TERRIBLE_YET_FAST_MODEL:
    warn("Will use a tiny model for speed - do not expect intelligence!")
    model_name = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit"
    # Qwen-2.5 should support up to 128k
    max_seq_length = 8192
    
    short_model_name = "qwen-2.5-coder-0.5b"
    chat_template = "qwen-2.5"
    core_training_args = {
        "sft": {
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "warmup_steps": 10,
            "num_train_epochs": 20,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
        },
        "lora": {
            "rank": 1,
            "alpha": 2,
        }
    }
else:
    model_name = "unsloth/Phi-4"
    # Immediately meaningful up to 16k for Phi4.
    max_seq_length = 16000
    
    short_model_name = "phi4"
    chat_template = "phi-4"
    core_training_args = {
        "sft": {
            "learning_rate": 2e-5,
            "weight_decay": 100,
            "warmup_steps": 5,
            "num_train_epochs": 80,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
        }
    }

# %%
model_name

# %%
from pathlib import Path
ROOT = Path("..")
RUN_DIR = ROOT / "run" / short_model_name
DATA_DIR = ROOT / "data"
SESSION_DIR = DATA_DIR / "sessions" / "validated"
OUTPUT_DATASET_PATH = RUN_DIR / "hf_dataset"
LORA_OUTPUT_PATH = RUN_DIR / "lora"

RUN_DIR.mkdir(exist_ok=True)

# %%
# %%time

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=core_training_args["lora"]["rank"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",],
    lora_alpha=core_training_args["lora"]["alpha"],
    lora_dropout=0, # 0 = fast path
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# See https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

# %%
# XML -> HF Dataset
from src.train.to_dataset import sessions_to_hf_dataset
sessions_to_hf_dataset(tokenizer, session_dir=SESSION_DIR, output_path=OUTPUT_DATASET_PATH)

# %% [markdown]
# ## Dataset Loading and Analysis

# %%
dataset = load_from_disk(OUTPUT_DATASET_PATH)
print(f"Loaded dataset with {len(dataset)} examples")

from src.train.utils import plot_token_distribution
plot_token_distribution(tokenizer, dataset)

# %% [markdown]
# ## Dataset Splitting

# %%
# Split if more than 1 sample
n_samples = len(dataset)
if n_samples > 1:
    eval_size = min(max(1, int(0.05 * n_samples)), n_samples - 1)
    splits = dataset.train_test_split(test_size=eval_size, seed=53)
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
if eval_dataset:
    print(eval_dataset['file'])
else:
    warn("No validation set")

# %%
train_dataset['text'][0][:100]

# %% [markdown]
# ## Training

# %%
# Configure training
config_args = {
    **core_training_args["sft"],
    "max_seq_length": max_seq_length,
    "dataset_num_proc": 1,
    "logging_steps": 1,
    "output_dir": "../run/model_training_outputs",
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "optim": "adamw_8bit",
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "report_to": "none",
    "packing": False,
}

# Add eval config if we have eval data
if eval_dataset is not None:
    config_args.update({
        "eval_strategy": "steps",
        "eval_steps": 1,
        #"save_strategy": "steps",
        #"save_steps": 5,
        #"load_best_model_at_end": True,
        #"metric_for_best_model": "loss",
        #"greater_is_better": False,
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
match chat_template:
    case "phi-4":
        # As per https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb
        print("Training on phi-4 responses")
        toro_kw = dict(
            instruction_part="<|im_start|>user<|im_sep|>",
            response_part="<|im_start|>assistant<|im_sep|>",
        )
    case "qwen-2.5" | "qwen-25" | "qwen25" | "qwen2.5":
        # As per https://colab.research.google.com/drive/18sN803sU23XuJV9Q8On2xgqHSer6-UZF?usp=sharing#scrollTo=juQiExuBG5Bt
        print("Training on Qwen-2.5 responses")
        toro_kw = dict(
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )

trainer = train_on_responses_only(
    trainer,
    **toro_kw
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
# %matplotlib widget
from src.train.utils import plot_training_loss
fig, ax, (train_line, eval_line) = plot_training_loss(trainer.state.log_history)

# %%
model.save_pretrained(LORA_OUTPUT_PATH)
tokenizer.save_pretrained(LORA_OUTPUT_PATH)

# %%
