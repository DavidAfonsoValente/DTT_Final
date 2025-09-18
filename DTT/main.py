import os
import sys
import argparse
import torch
import torch.nn as nn
import importlib
import importlib.util

from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset

from patch import patch_trainer_optimizer
from utils import *

# -----------------------------------------------------------------------------
# Vendored transformers override for custom GPT-2 (must occur BEFORE importing Auto*)
# -----------------------------------------------------------------------------
def install_custom_gpt2_if_available():
    """
    If ./custom_transformers/transformers/models/gpt2/modeling_gpt2.py exists,
    load it as a module and patch the live transformers module classes so that
    subsequent AutoModelForCausalLM.from_pretrained('gpt2') will instantiate
    the custom GPT2Model / GPT2LMHeadModel containing blend_* attributes.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    vendored_path = os.path.join(project_root, 'custom_transformers')
    expected_file = os.path.join(vendored_path, 'transformers', 'models', 'gpt2', 'modeling_gpt2.py')

    if not os.path.exists(vendored_path) or not os.path.exists(expected_file):
        return False

    # Ensure vendored path is first so any internal relative imports resolve there
    sys.path.insert(0, vendored_path)

    # Import base transformers first so we can patch its submodule attributes
    import transformers

    # Load custom modeling_gpt2 module explicitly
    spec = importlib.util.spec_from_file_location("transformers.models.gpt2.modeling_gpt2", expected_file)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    # Extract classes from custom module
    CustomGPT2Model = getattr(custom_module, 'GPT2Model', None)
    CustomGPT2LMHeadModel = getattr(custom_module, 'GPT2LMHeadModel', None)

    if CustomGPT2Model is None or CustomGPT2LMHeadModel is None:
        raise RuntimeError("Custom modeling_gpt2.py does not define GPT2Model/GPT2LMHeadModel")

    # Patch live transformers namespace
    transformers.models.gpt2.modeling_gpt2.GPT2Model = CustomGPT2Model
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel = CustomGPT2LMHeadModel

    # Do NOT reload transformers after patching; reloading can discard monkey-patches
    return True

installed_custom = install_custom_gpt2_if_available()

# Only import Auto classes AFTER patching so they bind to the patched module graph
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["WANDB_PROJECT"] = "latent-reasoning-gpt2"

def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    if dataset_name == "gsm8k":
        dataset = load_dataset('openai/gsm8k', 'main')
        return dataset[split].map(lambda batch: process_gsm8k(batch), batched=True,
                                  batch_size=chunk_size, load_from_cache_file=False)
    elif dataset_name == "prosqa":
        data_files = {
            'train': './data/prosqa_train.json',
            'validation': './data/prosqa_valid.json',
            'test': './data/prosqa_test.json'
        }
        dataset = load_dataset('json', data_files=data_files)
        return dataset[split].map(lambda batch: process_qa(batch), batched=True,
                                  batch_size=chunk_size, load_from_cache_file=False)
    elif dataset_name == "prontoqa":
        dataset = load_dataset("renma/ProntoQA")
        return dataset["train"].map(lambda batch: process_qa(batch), batched=True,
                                    batch_size=chunk_size, load_from_cache_file=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def assert_custom_blend_components(model):
    """
    Ensure the constructed model contains the custom attributes required by this script.
    Raise a helpful error if missing.
    """
    missing = []
    for attr in ["blend_lambda", "blend_gate_r", "blend"]:
        if not hasattr(model.transformer, attr):
            missing.append(attr)
    if missing:
        tip_lines = [
            "Custom GPT-2 blend components missing on model.transformer:",
            f"Missing: {', '.join(missing)}",
            "",
            "Troubleshooting tips:",
            "- Ensure custom_transformers/transformers/models/gpt2/modeling_gpt2.py exists and defines GPT2Model/GPT2LMHeadModel with blend_* attrs.",
            "- Make sure this script is run from the project root so the relative path resolves.",
            "- Do not reload transformers after monkey-patching; reloading can drop patches.",
            "- Consider clearing HF cache for GPT-2 if stale compiled modules interfere.",
        ]
        raise RuntimeError("\n".join(tip_lines))

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-{args.dataset}-group{args.group_size}"
                f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    os.makedirs(exp_name, exist_ok=True)
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Skipping...")
        return

    # Build model AFTER monkey-patch, and ensure remote code is not overriding
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=False,  # prevent remote overrides from bypassing our patch
    )

    # Verify custom fields exist; fail early with clear message
    assert_custom_blend_components(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reset lambda parameters (requires custom blend_lambda module to exist)
    model.transformer.blend_lambda.reset_lambda_parameters(
        r_min=args.residual_r_min, r_max=args.residual_r_max,
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["c_attn", "c_proj", "blend_gate_r", "blend_gate_i"],
        modules_to_save=["blend_lambda"],
    )
    model = get_peft_model(model, lora_config)

    training_args = GRPOConfig(
        learning_rate=args.lr,
        beta=args.beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        temperature=args.temperature,
        num_generations=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        save_steps=250,
        save_total_limit=3,
        report_to="wandb",
        output_dir=exp_name,
        gradient_checkpointing=True,
    )

    train_dataset = preprocess_dataset(args.dataset, 'train', chunk_size=500)
    eval_dataset = None
    if args.dataset == "prosqa":
        eval_dataset = preprocess_dataset(args.dataset, 'validation', chunk_size=500)

    process_answer_func = process_gsm8k_answer if args.dataset == "gsm8k" else process_qa_answer
    reward_func = get_reward_func(process_answer_func, efficiency_beta=args.efficiency_beta, is_math=args.dataset == "gsm8k")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
    )

    trainer.train()

    if args.dataset == "prosqa" and os.path.exists(f"./data/prosqa_test.json"):
        test_dataset = preprocess_dataset(args.dataset, 'test', chunk_size=500)
        print("Test evaluation:")
        trainer.evaluate(test_dataset)
    elif args.dataset == "prontoqa":
        pass  # No test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent reasoning model")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "prosqa", "prontoqa"])
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.0)
    parser.add_argument("--residual_r_max", type=float, default=1.0)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-3)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--efficiency_beta", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
