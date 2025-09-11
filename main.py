# main.py
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

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
        # Single split 'train', use for train
        return dataset["train"].map(lambda batch: process_qa(batch), batched=True, 
                                    batch_size=chunk_size, load_from_cache_file=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

class BlendLambda(nn.Module):
    c = 20.0  # Sharper transition for more binary-like blending

    def __init__(self, config):
        super().__init__()
        self.Lambda = nn.Parameter(torch.randn(config.n_embd))

    def reset_lambda_parameters(self, r_min=0.0, r_max=1.0):
        with torch.no_grad():
            nn.init.uniform_(self.Lambda, a=r_min, b=r_max)
            self.Lambda.data.copy_(
                -torch.log((self.Lambda ** (-1. / self.c)) - 1)
            )

    def forward(self, r_t):
        a_t = torch.exp(
            -self.c * nn.functional.softplus(-self.Lambda, beta=1, threshold=20) * r_t
        )
        return a_t

import types
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, BaseModelOutputWithPastAndCrossAttentions
import torch.nn.functional as F

def main(args):
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-{args.dataset}-group{args.group_size}"
                f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    os.makedirs(exp_name, exist_ok=True)
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Skipping...")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add custom modules to GPT2Model
    gpt2_model = model.model
    config = gpt2_model.config
    gpt2_model.blend_gate_r = nn.Linear(config.n_embd, config.n_embd)
    gpt2_model.blend_gate_i = nn.Linear(config.n_embd, config.n_embd)
    gpt2_model.blend_lambda = BlendLambda(config)
    gpt2_model.blend_lambda.reset_lambda_parameters(
        r_min=args.residual_r_min, r_max=args.residual_r_max,
    )

    # Add blend method
    def blend_method(self, embeds, residual, eps=1e-8):
        r_t = torch.sigmoid(self.blend_gate_r(embeds))
        i_t = torch.sigmoid(self.blend_gate_i(embeds))
        a_t = self.blend_lambda(r_t)
        blended = a_t * embeds + torch.sqrt(1 - a_t.pow(2) + eps) * (i_t * residual)
        return blended, a_t

    gpt2_model.blend = types.MethodType(blend_method, gpt2_model)

    # Patched forward: Fixed mask handling for GPT-2
    def patched_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = input_shape[0]
            seq_length = input_shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Set up position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, input_shape[-1], dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Prepare attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        else:
            attention_mask = None

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
            if attention_mask is not None and attention_mask.shape[-1] != seq_length + past_length:
                attention_mask = attention_mask[:, :, -seq_length:, :].contiguous()
            elif attention_mask is None:
                attention_mask = torch.full(
                    (batch_size, 1, seq_length, seq_length + past_length), 
                    torch.finfo(self.dtype).min, 
                    device=device, 
                    dtype=self.dtype
                )

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if past_key_values is None:
            # Full forward: Approximation - standard pass, shift hidden as residual, blend, re-pass
            # Temp standard forward
            temp_hidden_states = hidden_states
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            for i, layer in enumerate(self.h):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (temp_hidden_states,)
                layer_outputs = layer(
                    temp_hidden_states,
                    layer_past=None,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=False,
                    output_attentions=output_attentions,
                )
                temp_hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            # Shift as approx residual
            shifted_residual = torch.roll(temp_hidden_states, shifts=1, dims=1)
            shifted_residual[:, 0, :] = torch.zeros_like(shifted_residual[:, 0, :])

            # Blend
            blended_embeds, _ = self.blend(hidden_states, shifted_residual)

            # Re-run with blended
            hidden_states = blended_embeds
            presents = () if use_cache else None
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            for i, layer in enumerate(self.h):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(
                    hidden_states,
                    layer_past=None,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    presents = presents + (layer_outputs[1],)
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        else:
            # Incremental: Blend with zero residual (approx)
            zero_residual = torch.zeros_like(hidden_states)
            blended_embeds, _ = self.blend(hidden_states, zero_residual)
            hidden_states = blended_embeds

            # Standard incremental
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            presents = () if use_cache else None
            for i, (layer, layer_past) in enumerate(zip(self.h, past_key_values)):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    presents = presents + (layer_outputs[1],)
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            cross_attentions=all_self_attns,
        )

    gpt2_model.forward = types.MethodType(patched_forward, gpt2_model)

    # LoRA config for GPT-2
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["c_attn", "c_proj", "blend_gate_r", "blend_gate_i"],
        modules_to_save=["blend_lambda"],
        use_gradient_checkpointing=True,
        random_state=args.seed,
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
        evaluation_strategy="steps" if args.dataset in ["prosqa"] else "no",  # Eval only for ProsQA with val split
        eval_steps=250,
    )

    train_dataset = preprocess_dataset(args.dataset, 'train', chunk_size=500)
    eval_dataset = None
    if args.dataset == "prosqa":
        eval_dataset = preprocess_dataset(args.dataset, 'validation', chunk_size=500)
    # For ProntoQA, no val split, so no eval

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

    # Simple post-train eval on test if available
    if args.dataset == "prosqa" and os.path.exists(f"./data/prosqa_test.json"):
        test_dataset = preprocess_dataset(args.dataset, 'test', chunk_size=500)
        print("Test evaluation:")
        trainer.evaluate(test_dataset)
    elif args.dataset == "prontoqa":
        # No test, skip
        pass

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