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
            num_steps,
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
        self.num_steps = num_steps

    def stack(self, trajectory):
        for key, val in trajectory.items():
            trajectory[key] = torch.stack(val[:self.num_steps], dim=1).to(self.device)
        return trajectory

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

    def get_dataloader(self, seeds=[]):
        trajectories, metrics = self.preprocess()
    
        trajectories["value_target"] += trajectories["advantage"]
        metrics["unnorm advatange"] = trajectories["advantage"].mean().item()
        trajectories["advantage"] = (trajectories["advantage"] - trajectories["advantage"].mean()) / (trajectories["advantage"].std() + 1e-8)


        dataloader = DataLoader(
            list(
                zip(
                    trajectories["state"].float(),
                    trajectories["action"],
                    trajectories["reward"].float(),
                    trajectories["advantage"].float(),
                    trajectories["value_target"].float(),
                    trajectories["log_prob"].float(),
                )
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
        )

        return dataloader, metrics

    def preprocess(self):
        trajectory, pointers = self.get_trajectory()
        trajectory["next_value"] = trajectory["value"][1:]

        trajectory = self.stack(trajectory)
        trajectory["reward"] = trajectory["reward"].unsqueeze(-1)

        trajectory["advantage"] = torch.zeros_like(trajectory["reward"])
        trajectory["value_target"] = torch.zeros_like(trajectory["reward"])

        metrics = {}
        for i in pointers.keys():
            for left, right in pointers[i]:
                trajectory["advantage"][i, left:right] = torch.stack(
                    self.gae(
                        trajectory["value"][i, left:right],
                        trajectory["reward"][i, left:right],
                        trajectory["next_value"][i, left:right]
                    )
                )
                trajectory["value_target"][i, left:right] = torch.stack(
                    self.get_value_target(
                        trajectory["reward"][i, left:right]
                    )
                )

        metrics["rewards"] = trajectory["reward"].sum() / sum(map(len, pointers.values()))
        metrics["episode time"] = (len(pointers) * self.num_steps) / sum(map(len, pointers.values()))
        for key in trajectory:
            trajectory[key] = trajectory[key].reshape(
                trajectory[key].shape[0] * trajectory[key].shape[1], 
                -1
            )
        return trajectory, metrics

    def get_trajectory(self):
        raise NotImplementedError


class IsaacDataMiner(SimpleDataMiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_trajectory(self):
        self.model.eval()
        trajectory = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
            "log_prob": [],
        }

        observation = self.env.reset()["obs"]
        pointers = {key: [[0]] for key in range(observation.shape[0])}

        for i in range(self.num_steps + 1):
            action, log_prob, value = self.model.sample(observation)

            trajectory["state"].append(observation.clone().detach())
            trajectory["action"].append(action.clone().detach())
            trajectory["log_prob"].append(log_prob.clone().detach())
            trajectory["value"].append(value.clone().detach())

            observation, reward, terminated, info = self.env.step(action)
            observation = observation["obs"]
            trajectory["reward"].append(reward.clone().detach())

            for j in range(terminated.shape[0]):
                if terminated[j]:
                    pointers[j][-1].append(i + 1)
                    pointers[j].append([i + 1])

        for i in range(observation.shape[0]):
            if pointers[i][-1][0] == self.num_steps + 1:
                trajectory["value"][-1][i] = 0
                pointers[i].pop()
            else:
                pointers[i][-1].append(self.num_steps)
            if pointers[i][-1][0] == pointers[i][-1][1]:
                pointers[i].pop()

        return trajectory, pointers


class ClassicDataMiner(SimpleDataMiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_trajectory(self):
        self.model.eval()
        trajectory = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
            "log_prob": [],
        }

        observation, _ = self.env.reset()
        pointers = {key: [[0]] for key in range(observation.shape[0])}

        for i in range(self.num_steps + 1):
            action, log_prob, value = self.model.sample(observation)

            trajectory["state"].append(torch.tensor(observation))
            trajectory["action"].append(action.clone())
            trajectory["log_prob"].append(log_prob.clone())
            trajectory["value"].append(value.clone())

            if "cuda" in action.device.type:
                action = action.cpu()

            observation, reward, terminated, truncated, _ = self.env.step(action.numpy())
            terminated |= truncated

            trajectory["reward"].append(torch.tensor(reward))

            for j in range(terminated.shape[0]):
                if terminated[j]:
                    pointers[j][-1].append(i + 1)
                    pointers[j].append([i + 1])

        for i in range(observation.shape[0]):
            if pointers[i][-1][0] == self.num_steps + 1:
                trajectory["value"][-1][i] = 0
                pointers[i].pop()
            else:
                pointers[i][-1].append(self.num_steps)
            if pointers[i][-1][0] == pointers[i][-1][1]:
                pointers[i].pop()

        return trajectory, pointers
    





# class TrajectoryDataMiner(SimpleDataMiner):
#     def __init__(
#             self,
#             num_trajectories=2,
#             *args,
#             **kwargs
#         ):
#         super().__init__(*args, **kwargs)
#         self.num_trajectories = num_trajectories

#     def preprocess(self, seeds):
#         trajectories = []
#         rewards = []
#         episode_time = []
#         if not seeds:
#             for _ in range(self.num_trajectories):
#                 seeds.append(self.seed)
#                 self.seed += 1
#                 while self.seed in self.eval_seeds:
#                     self.seed += 1
    
#         for seed in seeds:
#             trajectory = self.get_trajectory(seed)
#             trajectories.append(trajectory)
#             rewards.append(sum(trajectory["reward"]))
#             episode_time.append(len(trajectory["reward"]))

#         trajectories = list_dict_extend(trajectories)
#         trajectories = stack_dict(trajectories)

#         return trajectories, {
#             "episode time": sum(episode_time) / len(episode_time),
#             "rewards": sum(rewards) / len(rewards),
#         }


# class StepsDataMiner(SimpleDataMiner):
#     def __init__(
#             self,
#             num_steps=5000,
#             *args,
#             **kwargs
#         ):
#         super().__init__(*args, **kwargs)
#         self.num_steps = num_steps
    
#     def preprocess(self, seeds):
#         trajectories = []
#         rewards = []
#         episode_time = []
    
#         for seed in seeds:
#             trajectory = self.get_trajectory(seed)
#             trajectories.append(trajectory)
#             rewards.append(sum(trajectory["reward"]))
#             episode_time.append(len(trajectory["reward"]))
#         else:
#             while sum(episode_time) < self.num_steps:
#                 trajectory = self.get_trajectory(self.seed)
#                 trajectories.append(trajectory)
#                 rewards.append(sum(trajectory["reward"]))
#                 episode_time.append(len(trajectory["reward"]))

#                 self.seed += 1
#                 while self.seed in self.eval_seeds:
#                     self.seed += 1

#         trajectories = list_dict_extend(trajectories)
#         trajectories = stack_dict(trajectories)

#         for key, val in trajectories.items():
#             trajectories[key] = val[:self.num_steps]

#         return trajectories, {
#             "episode time": sum(episode_time) / len(episode_time),
#             "rewards": sum(rewards) / len(rewards),
#         }