# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %pwd

# %%
import sys
sys.path.append('..')

# %%
import os
from src.persist.load import session_from_file
from src.preproc import session_to_chatml
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset

# %%
# Configuration
SESSION_DIR = "../data/sessions"  # Directory where your session XML files are saved
OUTPUT_DATASET_PATH = "../data/training_sessions_hf" # Path to save the Hugging Face dataset

# %%
# Load model and tokenizer (needed for chat template)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = 2048, # Or your desired max sequence length
    load_in_4bit = True,
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4", # Ensure chat template is correct for your model
)

# %%
def load_sessions_and_format(session_dir):
    """
    Loads session XML files from the specified directory,
    converts them to ChatML conversations, applies the chat template,
    and returns a list of dictionaries suitable for creating a HF dataset.
    """
    dataset_list = []
    session_files = [f for f in os.listdir(session_dir) if f.endswith(".xml")]

    if not session_files:
        print(f"No XML session files found in '{session_dir}'.")
        return None

    print(f"Found {len(session_files)} session files. Loading and formatting...")

    for session_file in session_files:
        session_path = os.path.join(session_dir, session_file)
        try:
            session = session_from_file(session_path)
            conversation = session_to_chatml(session)
            formatted_text = tokenizer.apply_chat_template(
                conversation, tokenize = False, add_generation_prompt = False
            )
            dataset_list.append({"conversations": conversation, "text": formatted_text}) # Store both for potential inspection
        except Exception as e:
            print(f"Error processing session file: {session_file}. Error: {e}")
            continue # Skip to the next file in case of error

    if not dataset_list:
        print("No sessions were successfully processed.")
        return None

    return dataset_list

# %%
# Load and format sessions
dataset_list = load_sessions_and_format(SESSION_DIR)

# Create Hugging Face Dataset if sessions were loaded successfully
if dataset_list:
    hf_dataset = Dataset.from_list(dataset_list)

    # Print a sample to verify
    print("\nSample dataset entry:")
    print(hf_dataset[0])

    # Save the dataset to disk
    hf_dataset.save_to_disk(OUTPUT_DATASET_PATH)
    print(f"\nHugging Face dataset saved to: {OUTPUT_DATASET_PATH}")
else:
    print("\nNo dataset created.")

# %% [markdown]
# Now you can load this dataset in your `train.py` notebook (or any training script) like this:
#
# ```python
# from datasets import load_from_disk
#
# training_dataset_hf = load_from_disk("data/training_sessions_hf")
#
# # Then use this dataset in your SFTTrainer:
# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = training_dataset_hf,
#     dataset_text_field = "text",
#     ...
# )
# ```

# %%
