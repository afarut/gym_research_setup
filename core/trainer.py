import gymnasium as gym
import torch
from common.utils import stack_dict, collate_to_device, list_dict_extend, reinforce_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from model.base import ModelBase


class BaseTrainer:
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError


class ValueReinforceTrainer(BaseTrainer):
    """Reinforce with value function (using GAE)"""
    def __init__(
            self,
            model: ModelBase,
            optimizer: torch.optim.Optimizer=None,
            entropy=False,
            entopy_coef=0.01,
            clip_value = 5,
        ):
        self.model = model
        self.optimizer = optimizer
        self.entropy = entropy
        if entropy:
            self.entopy_coef = entopy_coef
        else:
            self.entopy_coef = 0
        self.clip_value = clip_value
    
    def loss(self, state, action, reward, advantage, value_target):
        pred, entropy_loss, value = self.model.step(state, action, self.entropy)

        rl_loss = reinforce_loss(pred, advantage)
        value_loss = ((value_target - value) ** 2).mean()

        loss = rl_loss + value_loss + entropy_loss * self.entopy_coef

        return loss, {
            "loss": loss.item(),
            "rl_loss": rl_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def step(self, state, action, reward, advantage, value_target):
        loss, metrics = self.loss(state, action, reward, advantage, value_target)

        self.optimizer.zero_grad()

        loss.backward()
        clipped_value = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        metrics["grad_norm"] = clipped_value.item()

        self.optimizer.step()

        return metrics
