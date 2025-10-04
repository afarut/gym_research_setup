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


class TestDefaultTrainer:
    def __init__(self, model, logger, gym_name, batch_size=16, epochs=2000, gamma=0.95, alpha=0.8, entropy_weight=0.2, max_episode_steps=2000):
        self.env = gym.make(gym_name, max_episode_steps=max_episode_steps)
        self.model = model
        self.device = model.device
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=5e-5)
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.entropy_weight = entropy_weight
        self.epoch = 0

    def get_trajectory(self):
        self.model.eval()
        trajectory = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
        }
        observation, info = self.reset(seed=self.epoch)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, value = self.model.sample(observation)

            trajectory["state"].append(observation)
            trajectory["action"].append(action)
            trajectory["value"].append(value.detach())

            observation, reward, terminated, truncated, _ = self.env.step(action)
            
            trajectory["reward"].append(reward)
        self.logger.log({
            "episode time": len(trajectory["reward"]),
        })

        return trajectory

    def reset(self, seed=None):
        # Reset dataset +
        if seed is None:
            observation, info = self.env.reset()
        else:
            observation, info = self.env.reset(seed=seed)
        return observation, info

    def close(self):
        self.env.close()

    def step(self, state: torch.Tensor, action: torch.Tensor, entropy=False):
        assert len(action.shape) == 1

        entropy_loss = 0
        pred, value = self.model(state)
        value = value.squeeze()
        pred_actions = pred[torch.arange(pred.size(0)), action]
        if entropy:
            probs = F.softmax(pred, dim=-1)
            entropy_loss = (probs * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean()

        return pred_actions, entropy_loss, value

    def get_dataloader(self, state, action, reward, advantage, value_target):
        dataloader = DataLoader(
            list(
                zip(
                    state,
                    action,
                    reward,
                    advantage,
                    value_target
                )
            ), 
            batch_size=self.batch_size, 
            collate_fn=lambda x: collate_to_device(x, device=self.device),
            shuffle=True, 
            drop_last=False
        )
        return dataloader

    def gae(self, value, rewards, next_value):
        # detached_value - next value
        td_residual = (self.gamma * next_value.detach() + rewards.detach() - value.detach())
        result_rewards = [td_residual[-1]]
        for i in range(len(td_residual) - 2, -1, -1):
            result_rewards.append(result_rewards[-1] * self.gamma * self.alpha + td_residual[i])
        return torch.stack(result_rewards[::-1])

    def train_preprocess(self):
        trajectories = []
        rewards = []
        for _ in range(1):
            trajectory = self.get_trajectory()
            trajectories.append(trajectory)
            rewards.append(sum(trajectory["reward"]))

        rewards = np.array(rewards)


        trajectories = list_dict_extend(trajectories)
        trajectories = stack_dict(trajectories)
        self.logger.log(
            {
                "reward": rewards.mean(),
                "reward std": rewards.std(),
            }
        )

        # trajectories["reward"] = (trajectories["reward"] - trajectories["reward"].mean()) / (trajectories["reward"].std())
        return trajectories["state"], trajectories["action"], trajectories["reward"], trajectories["value"], torch.cat([trajectories["value"], torch.tensor([0]).to(self.device)])[1:, ]


    def get_value_target(self, rewards):
        result = [rewards[-1]]
        for i in range(len(rewards) - 2, -1, -1):
            result.append(result[-1] * self.gamma + rewards[i])
        return torch.stack(result[::-1])

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            dataset = {
                "states": [],
                "actions": [],
                "rewards": [],
                "advantage": [],
                "value_target": [],
            }
            for _ in range(1):
                self.epoch += 1
                states, actions, rewards, value, next_value = self.train_preprocess()
                advantage = self.gae(value, rewards, next_value)
                value_target = self.get_value_target(rewards) + advantage
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                # rewards = (rewards - rewards.mean(keepdim=True, dim=-1)) / rewards.std(keepdim=True, dim=-1)
                dataset["actions"].append(actions)
                dataset["states"].append(states)
                dataset["rewards"].append(rewards)
                dataset["advantage"].append(advantage)
                dataset["value_target"].append(value_target)
            dataloader = self.get_dataloader(
                torch.cat(dataset["states"]), 
                torch.cat(dataset["actions"]), 
                torch.cat(dataset["rewards"]),
                torch.cat(dataset["advantage"]),
                torch.cat(dataset["value_target"]),
            )
            self.model.train()
            rl_losses = 0
            all_loss = 0
            value_losses = 0

            for state, action, reward, advantage, value_target in dataloader:
                pred, entropy_loss, value = self.step(state, action, entropy=True)

                rl_loss = reinforce_loss(pred, advantage)
                value_loss = ((value_target - value) ** 2).mean()

                loss = rl_loss + value_loss#entropy_loss * (self.entropy_weight * (self.gamma ** self.epoch))
                all_loss += loss
                rl_losses += rl_loss
                value_losses += value_loss

                # all_loss /= len(actions)
                loss.backward()
                clip_value = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optimizer.step()
                self.optimizer.zero_grad()
            self.logger.log(
                {
                    "rl_losses": rl_losses / len(dataloader),
                    "value_loss": value_losses / len(dataloader),
                    # "grad norm": clip_value,
                }
            )



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
        metrics["grad_norm"] = clipped_value

        self.optimizer.step()

        return metrics
