import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from utils import format_sft_example

# --- Configuration ---
OUTPUT_FILE = "sft_dataset.jsonl"
BASE_MODEL_NAME = "gpt2" # We need the tokenizer to get the EOS token

# --- SFT Dataset Proportions ---
# Using a larger, more diverse dataset for a robust instruction model
NUM_GSM8K_EXAMPLES = 4000
NUM_PROSQA_EXAMPLES = 2500
NUM_PRONTOQA_EXAMPLES = 1500

def main():
    """
    Creates a blended SFT dataset, critically adding the EOS token to each example.
    """
    # Load the tokenizer specifically to get its end-of-sequence token
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.eos_token is None:
        # Some base models don't have an EOS token set, we'll use a common one
        tokenizer.eos_token = "<|endoftext|>"

    def format_and_add_eos(example):
        """Wrapper to format the text and append the crucial EOS token."""
        formatted_example = format_sft_example(example)
        formatted_example["text"] = formatted_example["text"] + tokenizer.eos_token
        return formatted_example

    # --- Load and format local datasets ---
    print("Loading and formatting local datasets...")

    gsm8k_files = {'train': './data/gsm_train.json'}
    prosqa_files = {'train': './data/prosqa_train.json'}
    prontoqa_files = {'train': './data/prontoqa_train.json'}

    gsm8k_dataset = load_dataset('json', data_files=gsm8k_files, split="train")
    gsm8k_dataset = gsm8k_dataset.shuffle(seed=42).select(range(NUM_GSM8K_EXAMPLES))
    gsm8k_dataset = gsm8k_dataset.map(format_and_add_eos, desc="Formatting GSM8K")

    prosqa_dataset = load_dataset('json', data_files=prosqa_files, split="train")
    prosqa_dataset = prosqa_dataset.shuffle(seed=42).select(range(NUM_PROSQA_EXAMPLES))
    prosqa_dataset = prosqa_dataset.map(format_and_add_eos, desc="Formatting ProSQA")

    prontoqa_dataset = load_dataset('json', data_files=prontoqa_files, split="train")
    prontoqa_dataset = prontoqa_dataset.shuffle(seed=42).select(range(NUM_PRONTOQA_EXAMPLES))
    prontoqa_dataset = prontoqa_dataset.map(format_and_add_eos, desc="Formatting ProntoQA")
    
    # --- Print one example to verify the EOS token is present ---
    print("\n--- Final SFT Example with EOS Token ---")
    print(prontoqa_dataset[0]['text'])
    print("----------------------------------------\n")

    # --- Combine and save the final dataset ---
    print("Combining and shuffling datasets...")
    final_dataset = concatenate_datasets([gsm8k_dataset, prosqa_dataset, prontoqa_dataset]).shuffle(seed=42)
    
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()