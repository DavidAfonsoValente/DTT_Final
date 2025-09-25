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
OUTPUT_DIR = "./gpt2-instruct-sft"
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 512
LR_BASE = 1e-5
LR_GATE = 1e-3  # if you have gating components
WARMUP_STEPS = 100
TRAIN_BATCH_SIZE = 8
ACCUM_STEPS = 4


def main():
    """Performs Supervised Fine-Tuning on the base GPT-2 model."""
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": [ANSWER_START]})
    print(f"Added new special token '{ANSWER_START}' to tokenizer vocabulary.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading SFT dataset from: {SFT_DATA_PATH}")
    raw_ds = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    print(f"Dataset size: {len(raw_ds)}")
    print("Sample record:", raw_ds[0])

    # Split for validation
    ds = raw_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        learning_rate=LR_BASE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting Supervised Fine-Tuning (SFT)...")
    trainer.train()

    print(f"\nSFT complete. Your new instruction-tuned model is saved to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
