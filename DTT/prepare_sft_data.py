import json
from datasets import load_dataset, concatenate_datasets

# This versatile system prompt teaches the model to be a general assistant
# but also reminds it of the special formatting for reasoning tasks.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instruction carefully and think step by step "
    "if the task is complex. For math or reasoning problems, provide the final answer after the #### tag."
)

OUTPUT_FILE = "sft_dataset.jsonl"
NUM_DOLLY_EXAMPLES = 4000  # Teaches general instruction following
NUM_GSM8K_EXAMPLES = 1000  # Teaches the specific CoT and "####" format

def format_dolly_example(example: dict) -> dict:
    """Formats a Dolly example into our desired single-text format."""
    if example["context"]:
        full_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User: {example['instruction']}\n\n"
            f"Context: {example['context']}\n\n"
            f"Assistant: {example['response']}"
        )
    else:
        full_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User: {example['instruction']}\n\n"
            f"Assistant: {example['response']}"
        )
    return {"text": full_text}


def format_gsm8k_example(example: dict) -> dict:
    """Formats a GSM8K example, which already contains CoT and the '####' tag."""
    full_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User: {example['question']}\n\n"
        f"Assistant: {example['answer']}"
    )
    return {"text": full_text}


def main():
    # --- Load and format Dolly dataset ---
    print(f"Loading {NUM_DOLLY_EXAMPLES} examples from databricks-dolly-15k...")
    dolly_dataset = load_dataset('databricks/databricks-dolly-15k', split="train")
    dolly_dataset = dolly_dataset.shuffle(seed=42).select(range(NUM_DOLLY_EXAMPLES))
    dolly_dataset = dolly_dataset.map(format_dolly_example, remove_columns=list(dolly_dataset.features))
    
    # --- Load and format GSM8K dataset ---
    print(f"Loading {NUM_GSM8K_EXAMPLES} examples from gsm8k...")
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main', split="train")
    gsm8k_dataset = gsm8k_dataset.shuffle(seed=42).select(range(NUM_GSM8K_EXAMPLES))
    gsm8k_dataset = gsm8k_dataset.map(format_gsm8k_example, remove_columns=list(gsm8k_dataset.features))

    # --- Combine and save the final dataset ---
    print("Combining datasets...")
    final_dataset = concatenate_datasets([dolly_dataset, gsm8k_dataset]).shuffle(seed=42)
    
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()