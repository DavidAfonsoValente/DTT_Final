import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# --- Configuration ---
BASE_MODEL = "gpt2"
SFT_DATA_PATH = "sft_dataset.jsonl"
OUTPUT_DIR = "./gpt2-instruct-sft" # Your new model will be saved here
NUM_EPOCHS = 6

def main():
    """
    Performs Supervised Fine-Tuning on the base GPT-2 model.
    """
    # --- 1. Load Model and Tokenizer ---
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # --- 2. Load and Prepare the Dataset ---
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

    # --- 3. Set Up Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    # --- 4. Create and Run the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Print some examples before training ---
    print("\n--- Training Data Examples ---")
    for i in range(2):
        print(f"--- Example {i+1} ---")
        example = tokenized_dataset[i]
        # Decode the tokenized input_ids back to text
        print(tokenizer.decode(example['input_ids']))
        print("---------------------\n")


    print("Starting Supervised Fine-Tuning (SFT)...")
    trainer.train()

    print(f"SFT complete. Your new instruction-tuned model is saved to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
