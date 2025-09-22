# patch.py

import types
from transformers.trainer import Trainer

def patch_trainer_optimizer(trainer, lr_residual_gate, lr_residual_Lambda):
    """
    Creates a custom optimizer that applies different learning rates to the
    standard model parameters and the special thinking_residual components.
    """
    def create_optimizer(self):
        opt_model = self.model_wrapped if hasattr(self, 'model_wrapped') else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # Use a dictionary to hold parameter groups for clarity
            param_groups = { "main_decay": [], "main_no_decay": [], "thinking_gate": [], "thinking_lambda": [] }

            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                
                if "thinking_residual_gate" in n:
                    param_groups["thinking_gate"].append(p)
                elif "thinking_residual_Lambda" in n:
                    param_groups["thinking_lambda"].append(p)
                elif n in decay_parameters:
                    param_groups["main_decay"].append(p)
                else:
                    param_groups["main_no_decay"].append(p)

            optimizer_grouped_parameters = [
                { "params": param_groups["main_decay"], "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay, },
                { "params": param_groups["main_no_decay"], "lr": self.args.learning_rate, "weight_decay": 0.0, },
                { "params": param_groups["thinking_gate"], "lr": lr_residual_gate, "weight_decay": self.args.weight_decay, },
                { "params": param_groups["thinking_lambda"], "lr": lr_residual_Lambda, "weight_decay": self.args.weight_decay, },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)