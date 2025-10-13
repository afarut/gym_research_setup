import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from common.utils import build_mlp


class AutoModel(nn.Module):
    def __init__(
        self,
        in_dim, 
        hidden_dim, 
        out_dim,
        device,
        discrete,
        body_depth,
        value_depth,
        head_depth,
        activation,
        clip_value=5,
        *args,
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.body_depth = body_depth
        self.value_depth = value_depth
        self.head_depth = head_depth
        self.discrete = discrete
        self.clip_value = clip_value

        if self.body_depth > 0:
            self.body = build_mlp(
                input_dim=self.in_dim, 
                hidden_dim=self.hidden_dim, 
                depth=self.body_depth,
                activation=activation,
            ).to(device)
        else:
            self.body = nn.Sequential()

        assert self.head_depth > 0
        self.head = build_mlp(
            input_dim=self.hidden_dim if self.body_depth > 0 else self.in_dim,
            hidden_dim=self.hidden_dim, 
            output_dim=self.out_dim * (int(not self.discrete) + 1),
            depth=self.head_depth,
            activation=activation,
        ).to(device)

        if self.value_depth > 0:
            self.value = build_mlp(
                input_dim=self.hidden_dim if self.body_depth > 0 else self.in_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                depth=self.value_depth,
                activation=activation,
            ).to(device)
        else:
            self.value = lambda x: (x.detach() * 0).mean(dim=-1)
    
        print("Body:", self.body)
        print("RL head:", self.head)
        print("Value head:", self.value)

    def clip(self):
        metrics = {}
        if self.body_depth > 0:
            clipped_value = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
            metrics["grad_norm"] = clipped_value.item()
        else:
            if self.value_depth > 0:
                clipped_value = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.clip_value)
                metrics["value_head_grad_norm"] = clipped_value.item()
            clipped_value = torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.clip_value)
            metrics["rl_head_grad_norm"] = clipped_value.item()
        return metrics

    def forward(self, x):
        logits = self.body(x)
        return self.head(logits), self.value(logits)

    def sample(self, state: torch.Tensor):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        assert len(state.shape) == 2 and state.shape[0] == 1 # batch_size == 1

        with torch.no_grad():
            pred, value = self.__call__(state)
        pred = pred.squeeze()
        value = value.squeeze()
        if self.discrete:
            distribution = dist.Categorical(logits=pred)
            result = distribution.sample().item()
        else:
            pred = pred.reshape(-1, 2)
            distribution = dist.Normal(loc=pred[..., 0], scale=(pred[..., 1] * 0.5).exp())
            result = distribution.sample().tolist()
        return result, value.item()

    def step(self, state: torch.Tensor, action: torch.Tensor):
        assert len(action.shape) == (1 + int(not self.discrete))
        batch_size = action.shape[0]

        pred, value = self.__call__(state)
        value = value.squeeze()

        if self.discrete:
            log_probs = pred.log_softmax(dim=-1)
            log_prob = log_probs[torch.arange(pred.size(0)), action]
            probs = F.softmax(pred, dim=-1)
            entropy_loss = (probs * log_probs).sum(dim=-1).mean()
        else:
            pred = pred.reshape(batch_size, -1, 2)
            distribution = dist.Normal(loc=pred[..., 0], scale=(pred[..., 1] * 0.5).exp())
            log_prob = distribution.log_prob(action)
            entropy_loss = -distribution.entropy().mean()

        return log_prob, entropy_loss, value