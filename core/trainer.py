import torch
from common.utils import reinforce_loss
from model.base import AutoModel


class BaseTrainer:
    def __init__(
            self,
            *,
            model: AutoModel,
            optimizer: torch.optim.Optimizer=None,
            entropy_coef=0.01,
        ):
        self.model = model
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef

    def step(self, **kwargs):
        loss, metrics = self.loss(**kwargs)

        self.optimizer.zero_grad()

        loss.backward()

        metrics.update(self.model.clip())

        self.optimizer.step()

        return metrics

    def loss(self, *args, **kwargs):
        raise NotImplementedError


class ReinforceTrainer(BaseTrainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def loss(self, state, action, advantage, *args, **kwargs):
        pred, entropy_loss, _ = self.model.step(state, action)

        rl_loss = -(pred * advantage).mean()

        loss = rl_loss + entropy_loss * self.entropy_coef

        return loss, {
            "loss": loss.item(),
            "rl_loss": rl_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }


class A2CTrainer(BaseTrainer):
    """Reinforce with Value function"""
    def __init__(
            self,
            *,
            value_coef,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.value_coef = value_coef

    def loss(self, state, action, advantage, value_target, *args, **kwargs):
        pred, entropy_loss, value = self.model.step(state, action)

        rl_loss = -(pred * advantage).mean()
        value_loss = ((value_target - value) ** 2).mean()

        loss = rl_loss + value_loss * self.value_coef + entropy_loss * self.entropy_coef

        return loss, {
            "loss": loss.item(),
            "rl_loss": rl_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }