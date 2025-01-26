import traceback
import os

from src.persist.load import session_from_file
from src.preproc import session_to_chatml
from datasets import Dataset

def load_sessions_and_format(tokenizer, session_dir):
    """
    Loads session XML files from the specified directory,
    converts them to ChatML conversations, applies the chat template from the given tokenizer (DO NOT mix model families carelessly!),
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
            dataset_list.append({"file": session_path, "conversations": conversation, "text": formatted_text})
        except Exception as e:
            print(f"Error processing session file: {session_file}. Error: {e}")
            traceback.print_exc()
            continue # Skip to the next file in case of error

    if not dataset_list:
        print("No sessions were successfully processed.")
        return None

    return dataset_list


def sessions_to_hf_dataset(tokenizer, session_dir, output_path):
    dataset_list = load_sessions_and_format(tokenizer, session_dir)

    if dataset_list:
        hf_dataset = Dataset.from_list(dataset_list)

        # Print a sample to verify
        #print("\nSample dataset entry:")
        #print(hf_dataset[0])

        # Save the dataset to disk
        hf_dataset.save_to_disk(output_path)
        print(f"\nHugging Face dataset saved to: {output_path}")
    else:
        print("\nNo dataset created.")
