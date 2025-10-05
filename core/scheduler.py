import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_steps=0, eta_min=0.0, last_epoch=-1):
        assert 0 <= warmup_steps < T_max

        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return [
                base_lr * step / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        progress = (step - self.warmup_steps) / max(1, self.T_max - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
