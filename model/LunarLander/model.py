import torch
from torch import nn
import torch.distributions as dist
from model.base import ModelBase


class LunarLander(ModelBase):
    def __init__(self, in_dim, hidden_dim, out_dim, device="cpu"):
        super().__init__(in_dim, hidden_dim, out_dim, device="cpu")

        print("***************")
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        ).to(device)
        self.value = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, x):
        logits = self.body(x)
        return self.head(logits), self.value(logits)