import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F


class ModelBase(nn.Module):
    def __init__(
        self,
        in_dim, 
        hidden_dim, 
        out_dim,
        device,
        *args,
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

    def forward(self, x):
        raise NotImplementedError

    def sample(self, state: torch.Tensor):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        assert len(state.shape) == 2 and state.shape[0] == 1

        pred, value = self.__call__(state)
        pred = pred.squeeze()
        value = value.squeeze()
        categorical = dist.Categorical(logits=pred)
        return categorical.sample().item(), value.item()

    def step(self, state: torch.Tensor, action: torch.Tensor, entropy=False):
        assert len(action.shape) == 1

        pred, value = self.__call__(state)
        value = value.squeeze()
        pred_action_logit = pred[torch.arange(pred.size(0)), action]

        probs = F.softmax(pred, dim=-1)
        entropy_loss = (probs * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean()
        if not entropy:
            entropy_loss = entropy_loss.detach()

        return pred_action_logit, entropy_loss, value