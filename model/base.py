import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from common.utils import build_mlp
from model.head import ContinuosHead, DiscreteHead


class AutoModel(nn.Module):
    def __init__(
        self,
        device,
        head,
        value=None,
        body=None,
        clip_value=1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.clip_value = clip_value
        self.head = head.to(self.device)

        if isinstance(self.head, DiscreteHead):
            self.distribution = dist.Categorical
        elif isinstance(self.head, ContinuosHead):
            self.distribution = dist.Normal
        else:
            raise ModuleNotFoundError("Distribution for your head not found")

        if body is None:
            self.body = nn.Sequential().to(self.device)
        else:
            self.body = body
        if value is None:
            self.value = lambda x: (x.detach() * 0).mean(dim=-1).unsqueeze(-1)
        else:
            self.value = value

        print("Body:", self.body)
        print("RL head:", self.head)
        print("Value head:", self.value)

    def clip(self):
        metrics = {}
        if self.body is not None:
            clipped_value = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
            metrics["grad_norm"] = clipped_value.item()
        else:
            if self.value is not None:
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
            state = torch.tensor(state, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            pred, value = self.__call__(state)

        distribution = self.distribution(**pred)
        result = distribution.sample()
        log_prob = distribution.log_prob(result)

        return result, log_prob.detach(), value

    def step(self, state: torch.Tensor, action: torch.Tensor):
        pred, value = self.__call__(state)

        distribution = self.distribution(**pred)
        if isinstance(distribution, dist.Categorical):
            log_prob = distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
        else:
            log_prob = distribution.log_prob(action)
        entropy_loss = -distribution.entropy().mean()

        return log_prob, entropy_loss, value