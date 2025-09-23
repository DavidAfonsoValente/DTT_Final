import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from utils import format_sft_example

# --- Configuration ---
OUTPUT_FILE = "sft_dataset.jsonl"
BASE_MODEL_NAME = "gpt2"
SFT_DATA_FILES = {
    'gsm8k': './data/gsm_train.json',
    'prosqa': './data/prosqa_train.json',
    'prontoqa': './data/prontoqa_train.json',
}
NUM_EXAMPLES = {
    'gsm8k': 200,
    'prosqa': 2000,
    'prontoqa': 2000,
}
# A new, unambiguous special token to signal the end of a generation
STOP_TOKEN = "<|stop|>"

def main():
    """
    Creates a blended SFT dataset from local files, adding a custom stop token
    to each example to teach the model when to terminate generation.
    """
    # --- 1. Load tokenizer and add the new special token ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Add our new, unambiguous stop token
    tokenizer.add_special_tokens({'additional_special_tokens': [STOP_TOKEN]})
    
    print(f"Added new special token '{STOP_TOKEN}' to tokenizer.")

    def format_and_add_stop_token(example):
        """Wrapper to format the text and append the crucial stop token."""
        formatted_example = format_sft_example(example)
        formatted_example["text"] += STOP_TOKEN
        return formatted_example

    all_datasets = []
    for name, path in SFT_DATA_FILES.items():
        print(f"Loading {NUM_EXAMPLES[name]} examples from {path}...")
        dataset = load_dataset('json', data_files={'train': path}, split="train")
        dataset = dataset.shuffle(seed=42).select(range(NUM_EXAMPLES[name]))
        all_datasets.append(dataset)

    # --- 2. Combine datasets and apply formatting ---
    print("Combining and formatting datasets...")
    final_dataset = concatenate_datasets(all_datasets).shuffle(seed=42)
    final_dataset = final_dataset.map(format_and_add_stop_token, desc="Formatting and adding stop token")

    # --- 3. Print one example to verify the format ---
    print("\n--- Final SFT Example with Stop Token ---")
    print(repr(final_dataset[0]['text'])) # Use repr to clearly see the special token
    print("---------------------------------------\n")

    # --- 4. Save the final dataset ---
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()