import os
import sys
import argparse
import torch
from datasets import load_dataset, concatenate_datasets

# --- Clean method to use your local libraries ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning-final"

# --- START: Corrected Custom Stopping Criteria ---
# This class stops generation on the newline character AFTER '####' has been seen.
class StopOnAnswerCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.answer_sequence_ids = tokenizer.encode("####", add_special_tokens=False)
        self.newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.answer_started = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Get the generated sequence as a list of token IDs
        sequence = input_ids[0].tolist()

        # Check if the '####' sequence is present
        if not self.answer_started:
            # A simple way to check for a subsequence
            for i in range(len(sequence) - len(self.answer_sequence_ids) + 1):
                if sequence[i:i+len(self.answer_sequence_ids)] == self.answer_sequence_ids:
                    self.answer_started = True
                    break
        
        # If '####' has been seen, stop at the next newline character
        if self.answer_started:
            if sequence[-1] == self.newline_token_id:
                # Reset for the next generation in the batch
                self.answer_started = False
                return True
                
        return False

def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-{args.dataset}-group{args.group_size}"
                f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    os.makedirs(exp_name, exist_ok=True)
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Skipping...")
        return

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    model.answer_start = ANSWER_START
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["c_attn", "c_proj"],
        modules_to_save=["thinking_residual_gate_r", "thinking_residual_gate_i", "thinking_residual_Lambda", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    
    model.base_model.model.transformer.thinking_residual_Lambda.reset_lambda_parameters(
        r_min=args.residual_r_min, r_max=args.residual_r_max,
    )
    model.print_trainable_parameters()

    # --- Create the stopping criteria instance ---
    stopping_criteria = StoppingCriteriaList([StopOnAnswerCriteria(tokenizer)])

    training_args = GRPOConfig(
        learning_rate=args.lr, beta=args.beta, adam_beta1=0.9, adam_beta2=0.99,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type, optim=args.optimizer,
        max_grad_norm=args.max_grad_norm, logging_steps=1, bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(), temperature=args.temperature,
        num_generations=args.group_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        num_train_epochs=1, save_steps=250, save_total_limit=3,
        report_to="wandb", output_dir=exp_name, gradient_checkpointing=True,
    )

    # --- Load and process the correct dataset for RL ---
    print(f"Loading RL dataset: {args.dataset}")
    if args.dataset == "all":
        gsm_train = load_dataset('json', data_files={'train': './data/gsm_train.json'}, split="train").map(process_rl_batch, batched=True)
        prosqa_train = load_dataset('json', data_files={'train': './data/prosqa_train.json'}, split="train").map(process_rl_batch, batched=True)
        prontoqa_train = load_dataset('json', data_files={'train': './data/prontoqa_train.json'}, split="train").map(process_rl_batch, batched=True)
        train_dataset = concatenate_datasets([gsm_train, prosqa_train, prontoqa_train]).shuffle(seed=42)
    else:
        train_dataset = load_dataset('json', data_files={'train': f'./data/{args.dataset}_train.json'}, split="train")
        train_dataset = train_dataset.map(process_rl_batch, batched=True)

    is_math = args.dataset in ["gsm8k", "all"]
    process_answer_func = process_math_answer if is_math else process_qa_answer
    reward_func = get_reward_func(process_answer_func, efficiency_beta=args.efficiency_beta)
    
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args, train_dataset=train_dataset,
        peft_config=lora_config,
        # --- Pass the stopping criteria to the trainer ---
        stopping_criteria=stopping_criteria,
    )
    
    patch_trainer_optimizer(trainer, args.lr_residual_gate, args.lr_residual_Lambda)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model with HRPO")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "prosqa", "prontoqa", "all"])
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.99, help="Start high for gentle start")
    parser.add_argument("--residual_r_max", type=float, default=0.999, help="Start high for gentle start")
    parser.add_argument("--lr_residual_gate", type=float, default=1e-3)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--efficiency_beta", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0, help="High temperature for exploration")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="./gpt2-instruct-sft", help="Model to train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)