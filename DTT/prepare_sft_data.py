import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from utils import format_sft_example, ANSWER_START

# --- Configuration ---
OUTPUT_FILE = "sft_dataset.jsonl"
BASE_MODEL_NAME = "gpt2"
SFT_DATA_FILES = {
    'gsm8k': './data/gsm_train.json',
    'prosqa': './data/prosqa_train.json',
    'prontoqa': './data/prontoqa_train.json',
}
NUM_EXAMPLES = {
    'gsm8k': 0,
    'prosqa': 20000,
    'prontoqa': 0,
}

def main():
    """
    Creates a blended SFT dataset, adding both the ANSWER_START tag and the
    model's default EOS token to each example.
    """
    # --- 1. Load tokenizer and add the ANSWER_START as a new special token ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token  # Define here for use in the formatting function
    
    # Add ANSWER_START as a new special token. This helps the model treat it as a single unit.
    tokenizer.add_special_tokens({'additional_special_tokens': [ANSWER_START]})
    print(f"Added new special token '{ANSWER_START}' to tokenizer vocabulary.")
    
    def format_and_add_eos_token(example: dict) -> dict:
        """Wrapper to format the text and append the crucial EOS token."""
        formatted_example = format_sft_example(example)
        formatted_example["text"] += eos_token
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
    final_dataset = final_dataset.map(format_and_add_eos_token, desc="Formatting and adding EOS token")

    # --- 3. Print one example to verify the format ---
    print(f"\n--- Final SFT Example (verifying format with '{ANSWER_START}' and EOS Token) ---")
    print(repr(final_dataset[0]['text']))  # Use repr to clearly see the special token
    print("--------------------------------------------------------------------------\n")

    # --- 4. Save the final dataset ---
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()