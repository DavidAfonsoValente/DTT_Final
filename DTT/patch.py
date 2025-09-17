import types
from transformers.trainer import Trainer

def patch_trainer_optimizer(trainer, lr_blend_gate=1e-3, lr_blend_lambda=1e-3):
    def create_optimizer(self):
        opt_model = self.model_wrapped if hasattr(self, 'model_wrapped') else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if ("blend" not in n and n in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if ("blend" not in n and n not in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if ("blend_gate" in n and p.requires_grad)
                    ],
                    "lr": lr_blend_gate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if ("blend_lambda" in n and p.requires_grad)
                    ],
                    "lr": lr_blend_lambda,
                    "weight_decay": self.args.weight_decay,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                from torch.optim import AdamW
                optimizer_cls = AdamW
                optimizer_kwargs = {}

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    trainer._old_create_optimizer = trainer.create_optimizer
    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)