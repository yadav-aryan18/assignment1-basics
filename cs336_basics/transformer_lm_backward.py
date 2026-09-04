from __future__ import annotations

import math
import torch

from typing import Optional
from collections.abc import Iterable, Callable




def crossEntropyLoss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.view(-1, preds.size(-1))
    targets = targets.view(-1)

    preds_max, _ = preds.max(dim=-1, keepdim=True)
    preds = preds - preds_max

    expPreds = torch.exp(preds)
    sumExPreds = expPreds.sum(dim=-1, keepdim=True)
    sumExPreds = torch.log(sumExPreds)

    preds = sumExPreds - preds

    preds = torch.gather(preds, dim=-1, index=targets.unsqueeze(-1))
    return preds.mean()



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                lr_t = lr * math.sqrt(1 - (beta2)**t)/(1 - (beta1)**t)

                grad = p.grad.data
                p.data -= lr * weight_decay * p.data

                first_moment = state.setdefault("first_moment", torch.zeros_like(p.data))
                second_moment = state.setdefault("second_moment", torch.zeros_like(p.data))

                first_moment = first_moment * beta1 + (1 - beta1) * grad
                second_moment = second_moment * beta2 + (1 - beta2) * (grad)**2

                p.data -= lr_t * (first_moment / (torch.sqrt(second_moment) + eps))
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment
                state["t"] = t + 1
        return loss


def cosineAnnealingScheduler(
        max_learning_rate: float,
        min_learning_rate: float,
        current_step: int,
        warmup_steps: int,
        final_steps: int
    ) -> float:

        if current_step < warmup_steps:
            learning_rate = current_step * max_learning_rate / warmup_steps
        elif warmup_steps <= current_step <= final_steps:
            learning_rate = (                    min_learning_rate
                                                        +
                0.5 * (1 + math.cos((current_step - warmup_steps) * (math.pi) / (final_steps - warmup_steps)))
                                                        *
                                    (max_learning_rate - min_learning_rate)
                            )
        else:
            learning_rate = min_learning_rate
        return learning_rate


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    # total_params = torch.concat([p.grad.data for p in params if p.grad is not None and p.requires_grad])
    total_squared = 0.0
    for p in params:
        if p.grad is not None and p.requires_grad:
            total_squared += torch.sum(p.grad.data * p.grad.data).item()
    total_squared = max(total_squared, 0.0)
    grad_norm = math.sqrt(total_squared)
    if grad_norm < max_norm:
        return
    scaling_factor = max_norm / (grad_norm + eps)
    for p in params:
        if p.grad is not None and p.requires_grad:
            p.grad.data *= scaling_factor
