import json
from datasets import load_dataset, concatenate_datasets
# This imports the consistent formatting functions from your new utils.py
from utils import format_dolly_sft, format_gsm8k_sft

OUTPUT_FILE = "sft_dataset.jsonl"
NUM_DOLLY_EXAMPLES = 4000  # Teaches general instruction-following
NUM_GSM8K_EXAMPLES = 1000  # Teaches the specific math CoT format

def main():
    """
    Creates a blended dataset for SFT by combining general instruction
    examples (Dolly) with task-specific reasoning examples (GSM8K).
    """
    # --- Load and format Dolly dataset ---
    print(f"Loading {NUM_DOLLY_EXAMPLES} examples from databricks-dolly-15k...")
    dolly_dataset = load_dataset('databricks/databricks-dolly-15k', split="train")
    dolly_dataset = dolly_dataset.shuffle(seed=42).select(range(NUM_DOLLY_EXAMPLES))
    dolly_dataset = dolly_dataset.map(format_dolly_sft, remove_columns=list(dolly_dataset.features))
    
    # --- Load and format GSM8K dataset ---
    print(f"Loading {NUM_GSM8K_EXAMPLES} examples from gsm8k...")
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split="train")
    gsm8k_dataset = gsm8k_dataset.shuffle(seed=42).select(range(NUM_GSM8K_EXAMPLES))
    gsm8k_dataset = gsm8k_dataset.map(format_gsm8k_sft, remove_columns=list(gsm8k_dataset.features))

    # --- Combine and save the final dataset ---
    print("Combining and shuffling datasets...")
    final_dataset = concatenate_datasets([dolly_dataset, gsm8k_dataset]).shuffle(seed=42)
    
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()
