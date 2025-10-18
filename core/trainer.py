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

    def loss(self, state, action, advantage, log_prob, *args, **kwargs):
        new_log_prob, entropy_loss, _ = self.model.step(state, action)

        rl_loss = -(new_log_prob * advantage).mean()

        loss = rl_loss + entropy_loss * self.entropy_coef

        ratio = (new_log_prob - log_prob).exp()

        return loss, {
            "loss": loss.item(),
            "rl_loss": rl_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": (ratio - 1).pow(2).mean().item(),
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

    def loss(self, state, action, advantage, value_target, log_prob, *args, **kwargs):
        new_log_prob, entropy_loss, value = self.model.step(state, action)

        rl_loss = -(new_log_prob * advantage).mean()
        value_loss = ((value_target - value) ** 2).mean()

        loss = rl_loss + value_loss * self.value_coef + entropy_loss * self.entropy_coef

        ratio = (new_log_prob - log_prob).exp()

        return loss, {
            "loss": loss.item(),
            "rl_loss": rl_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": (ratio - 1).pow(2).mean().item(),
        }


class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization"""
    def __init__(
            self,
            *,
            value_coef=1,
            clip_param=0.2,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.value_coef = value_coef
        self.clip_param = clip_param

    def loss(self, state, action, advantage, value_target, log_prob, *args, **kwargs):        
        new_log_prob, entropy_loss, value = self.model.step(state, action)

        ratio = (new_log_prob - log_prob).exp()
        
        surr1 = ratio * advantage
        
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = ((value_target - value) ** 2).mean()

        loss = policy_loss + value_loss * self.value_coef + entropy_loss * self.entropy_coef # Предполагая, что self.entropy_coef определен

        return loss, {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(), # Переименовано из "rl_loss"
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": (ratio - 1).pow(2).mean().item(), # Полезно для мониторинга
        }