import json
from datasets import load_dataset, concatenate_datasets
from utils import SYSTEM_PROMPT, ANSWER_START

OUTPUT_FILE = "sft_dataset.jsonl"
# Adjust how many examples from each dataset to use for SFT
NUM_GSM8K_EXAMPLES = 2500
NUM_PROSQA_EXAMPLES = 1500
NUM_PRONTOQA_EXAMPLES = 1000

def format_sft_example(example: dict) -> dict:
    """
    Formats an example for SFT. It combines the 'steps' and 'answer' fields
    into a single, high-quality assistant response, cleaning up any special characters.
    It correctly handles cases where 'steps' is a list of strings.
    """
    question = example.get('question', '')
    steps = example.get('steps', '')
    answer = example.get('answer', '')

    # --- FIX: Correctly process list-based steps ---
    # Check if steps is a list and join it into a multi-line string,
    # while also cleaning each step.
    if isinstance(steps, list):
        # Remove any extra characters from each step for cleaner training data
        cleaned_steps = [step.strip().replace("<<", "").replace(">>", "") for step in steps]
        steps = "\n".join(cleaned_steps)
    # --- END OF FIX ---
    
    # Construct the perfect assistant response in Chain-of-Thought format
    assistant_response = f"{steps}\n{ANSWER_START} {answer}"
    full_text = f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant: {assistant_response}"
    return {"text": full_text}


def main():
    """
    Creates a blended dataset for SFT by combining your three specified datasets,
    all loaded from local files and using a single, unified data structure.
    """
    # --- Load and format GSM8K dataset from local file ---
    print(f"Loading {NUM_GSM8K_EXAMPLES} examples from local gsm8k data...")
    gsm8k_files = {'train': './data/gsm_train.json'}
    gsm8k_dataset = load_dataset('json', data_files=gsm8k_files, split="train")
    gsm8k_dataset = gsm8k_dataset.shuffle(seed=42).select(range(NUM_GSM8K_EXAMPLES))
    gsm8k_dataset = gsm8k_dataset.map(format_sft_example, desc="Formatting GSM8K")
    print("\n--- GSM8K SFT Example ---")
    print(gsm8k_dataset[0]['text'])
    print("------------------------\n")

    # --- Load and format ProSQA dataset from local file ---
    print(f"Loading {NUM_PROSQA_EXAMPLES} examples from local ProSQA data...")
    prosqa_files = {'train': './data/prosqa_train.json'}
    prosqa_dataset = load_dataset('json', data_files=prosqa_files, split="train")
    prosqa_dataset = prosqa_dataset.shuffle(seed=42).select(range(NUM_PROSQA_EXAMPLES))
    prosqa_dataset = prosqa_dataset.map(format_sft_example, desc="Formatting ProSQA")
    print("\n--- ProSQA SFT Example ---")
    print(prosqa_dataset[0]['text'])
    print("-------------------------\n")

    # --- Load and format ProntoQA dataset from local file ---
    print(f"Loading {NUM_PRONTOQA_EXAMPLES} examples from local ProntoQA data...")
    prontoqa_files = {'train': './data/prontoqa_train.json'}
    prontoqa_dataset = load_dataset("json", data_files=prontoqa_files, split="train")
    prontoqa_dataset = prontoqa_dataset.shuffle(seed=42).select(range(NUM_PRONTOQA_EXAMPLES))
    prontoqa_dataset = prontoqa_dataset.map(format_sft_example, desc="Formatting ProntoQA")
    print("\n--- ProntoQA SFT Example ---")
    print(prontoqa_dataset[0]['text'])
    print("---------------------------\n")

    # --- Combine and save the final dataset ---
    print("Combining and shuffling datasets...")
    final_dataset = concatenate_datasets([gsm8k_dataset, prosqa_dataset, prontoqa_dataset]).shuffle(seed=42)
    
    print(f"Saving {len(final_dataset)} examples to {OUTPUT_FILE}...")
    final_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
            
    print("SFT dataset created successfully!")

if __name__ == "__main__":
    main()
