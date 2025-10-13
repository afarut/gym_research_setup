import torch
import numpy as np
from torch.utils.data import DataLoader
from model.base import AutoModel
from gymnasium import Env
from common.utils import stack_dict, collate_to_device, list_dict_extend, reinforce_loss


class SimpleDataMiner:
    def __init__(
            self, 
            model: AutoModel, 
            env: Env,
            batch_size,
            eval_seeds,
            start_seed=0,
            alpha=0.95,
            gamma=0.99,
            device="cpu",
            drop_last=False,
        ):
        assert batch_size > 1
        self.model = model
        self.env = env
        self.seed = start_seed
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.eval_seeds = eval_seeds
        self.drop_last = drop_last

    def gae(self, value: list, rewards: list, next_value: list) -> list:
        td_residual = []
        for i in range(len(value)):
            td_residual.append(self.gamma * next_value[i] + rewards[i] - value[i])

        result_rewards = [td_residual[-1]]
        for i in range(len(td_residual) - 2, -1, -1):
            result_rewards.append(
                result_rewards[-1] * self.gamma * self.alpha + td_residual[i]
            )
        return result_rewards[::-1]

    def get_value_target(self, rewards):
        result = [rewards[-1]]
        for i in range(len(rewards) - 2, -1, -1):
            result.append(result[-1] * self.gamma + rewards[i])
        return result[::-1]
    
    def get_trajectory(self, seed):
        self.model.eval()
        trajectory = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
        }
        observation, info = self.env.reset(seed=seed)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, value = self.model.sample(observation)

            trajectory["state"].append(observation.astype(np.float32))
            trajectory["action"].append(action)
            trajectory["value"].append(value)

            observation, reward, terminated, truncated, _ = self.env.step(action)

            trajectory["reward"].append(reward)

        self.env.close()
        trajectory["next_value"] = trajectory["value"].copy() + [0]
        trajectory["next_value"].pop(0)
        trajectory["advantage"] = self.gae(
            trajectory["value"],
            trajectory["reward"],
            trajectory["next_value"]
        )
        trajectory["value_target"] = self.get_value_target(trajectory["reward"])
        return trajectory
    
    def preprocess(self, seeds):
        raise NotImplementedError

    def get_dataloader(self, seeds=[]):
        trajectories, metrics = self.preprocess(seeds)
    
        trajectories["value_target"] += trajectories["advantage"]
        trajectories["advantage"] = (trajectories["advantage"] - trajectories["advantage"].mean()) / (trajectories["advantage"].std() + 1e-8)

        dataloader = DataLoader(
            list(
                zip(
                    trajectories["state"],
                    trajectories["action"],
                    trajectories["reward"],
                    trajectories["advantage"],
                    trajectories["value_target"]
                )
            ),
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_to_device(x, device=self.device),
            shuffle=True,
            drop_last=self.drop_last
        )

        return dataloader, metrics


class TrajectoryDataMiner(SimpleDataMiner):
    def __init__(
            self,
            num_trajectories=2,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.num_trajectories = num_trajectories
    
    def preprocess(self, seeds):
        trajectories = []
        rewards = []
        episode_time = []
        if not seeds:
            for _ in range(self.num_trajectories):
                seeds.append(self.seed)
                self.seed += 1
                while self.seed in self.eval_seeds:
                    self.seed += 1
    
        for seed in seeds:
            trajectory = self.get_trajectory(seed)
            trajectories.append(trajectory)
            rewards.append(sum(trajectory["reward"]))
            episode_time.append(len(trajectory["reward"]))

        trajectories = list_dict_extend(trajectories)
        trajectories = stack_dict(trajectories)

        return trajectories, {
            "episode time": sum(episode_time) / len(episode_time),
            "rewards": sum(rewards) / len(rewards),
        }


class StepsDataMiner(SimpleDataMiner):
    def __init__(
            self,
            num_steps=5000,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
    
    def preprocess(self, seeds):
        trajectories = []
        rewards = []
        episode_time = []
    
        for seed in seeds:
            trajectory = self.get_trajectory(seed)
            trajectories.append(trajectory)
            rewards.append(sum(trajectory["reward"]))
            episode_time.append(len(trajectory["reward"]))
        else:
            while sum(episode_time) < self.num_steps:
                trajectory = self.get_trajectory(self.seed)
                trajectories.append(trajectory)
                rewards.append(sum(trajectory["reward"]))
                episode_time.append(len(trajectory["reward"]))

                self.seed += 1
                while self.seed in self.eval_seeds:
                    self.seed += 1

        trajectories = list_dict_extend(trajectories)
        trajectories = stack_dict(trajectories)

        for key, val in trajectories.items():
            trajectories[key] = val[:self.num_steps]

        return trajectories, {
            "episode time": sum(episode_time) / len(episode_time),
            "rewards": sum(rewards) / len(rewards),
        }