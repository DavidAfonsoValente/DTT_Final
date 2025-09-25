import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from utils import ANSWER_START  # Import ANSWER_START from utils for consistency

# --- Configuration ---
BASE_MODEL = "gpt2"
SFT_DATA_PATH = "sft_dataset.jsonl"
OUTPUT_DIR = "./gpt2-instruct-sft" # Your new model will be saved here
NUM_EPOCHS = 5

def main():
    """Performs Supervised Fine-Tuning on the base GPT-2 model."""
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"  # Set to left padding for consistency with other scripts
    tokenizer.add_special_tokens({'additional_special_tokens': [ANSWER_START]})
    print(f"Added new special token '{ANSWER_START}' to tokenizer vocabulary.")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for the new special token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Loading SFT dataset from: {SFT_DATA_PATH}")
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting Supervised Fine-Tuning (SFT)...")
    trainer.train()

    print(f"\nSFT complete. Your new instruction-tuned model is saved to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()