import torch
from torch import nn
from common.utils import build_mlp


class ContinuosHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, depth, sub_head_depth=1):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.activation = activation
        self.depth = depth
        self.sub_head_depth = sub_head_depth

        self.head = build_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=None,
            depth=self.depth,
            activation=activation,
        )

        self.scale_head = build_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            depth=self.sub_head_depth,
            activation=activation,
        )

        self.loc_head = build_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            depth=self.sub_head_depth,
            activation=activation,
        )

    def forward(self, x):
        logits = self.head(x)
        return {"loc": self.loc_head(logits), "scale": torch.log(1 + torch.exp(self.scale_head(logits))) + 1e-5}


class DiscreteHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, depth):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.activation = activation
        self.depth = depth

        self.head = build_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            activation=activation,
        )

    def forward(self, x):
        return {"logits": self.head(x)}