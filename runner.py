import torch
import numpy as np
from model.base import ModelBase
from gymnasium import Env
from hydra.utils import instantiate
from logger.base import LoggerBase
from trainer import BaseTrainer
from core.dataminer import SimpleDataMiner
from tqdm import tqdm
from common.utils import stack_dict, list_dict_extend, add_meta, rand_by_time, restart_tensorboard
from core.checkpoint import CheckPointer


class Runner:
    def __init__(
            self, 
            model: ModelBase, 
            env: Env,
            logger: LoggerBase,
            trainer: BaseTrainer,
            data_miner: SimpleDataMiner,
            num_pseudo_epochs: int,
            eval_freq: int,
            tensorboard_port: int,
            name=None,
            checkpoint_path=None,
            checkpoint_drop_past=False,
            *args,
            **kwargs
        ):
        assert name is not None

        np.random.seed(42)
        torch.manual_seed(42)

        self.model = instantiate(model)
        self.model.train()

        optimizer = instantiate(trainer.optimizer, self.model.parameters())

        self.env = instantiate(env)
        self.logger = instantiate(logger)

        add_meta({
            "run_id": self.logger.log_dir + "/tensorboard",
            "alias": name,
            "visible": True
        })

        restart_tensorboard(tensorboard_port)

        self.trainer = instantiate(trainer, model=self.model, optimizer=optimizer)
        self.data_miner = instantiate(data_miner, model=self.model, env=self.env)

        self.num_pseudo_epochs = num_pseudo_epochs
        self.eval_freq = eval_freq

        self.checkpointer = CheckPointer(
            self.model, 
            optimizer, 
            self.env, 
            self.logger, 
            self.data_miner, 
            drop_past=checkpoint_drop_past
        )
        self.checkpoint_epochs = 0
        if checkpoint_path is not None:
            self.checkpoint_epochs = self.checkpointer.load_checkpoint(checkpoint_path, "last")
        assert self.num_pseudo_epochs - self.checkpoint_epochs > 0

    def train(self):
        for epoch in tqdm(range(self.checkpoint_epochs, self.num_pseudo_epochs)):
            if epoch % self.eval_freq == 0:
                metrics = self.eval()
                self.checkpointer.save(epoch, metrics)
                self.model.train()

            dataloader, metrics = self.data_miner.get_dataloader()
            self.logger.log(metrics, prefix="train/")

            rollout_metrics = []

            for state, action, reward, advantage, value_target in dataloader:
                metrics = self.trainer.step(state, action, reward, advantage, value_target)
                metrics = {key: [val] for key, val in metrics.items()}
                rollout_metrics.append(metrics)

            rollout_metrics = stack_dict(
                list_dict_extend(
                    rollout_metrics
                )
            )
            self.logger.log({key: val.mean() for key, val in rollout_metrics.items()}, prefix="train/")

        self.checkpointer.save_checkpoint(self.num_pseudo_epochs, "last")
        self.logger.close()


    def eval(self):
        self.model.eval()
        dataloader, trajectory_metrics = self.data_miner.get_dataloader(self.data_miner.eval_seeds)
        self.logger.log(trajectory_metrics, prefix="eval/")

        rollout_metrics = []

        for state, action, reward, advantage, value_target in dataloader:
            _, metrics = self.trainer.loss(state, action, reward, advantage, value_target)
            metrics = {key: [val] for key, val in metrics.items()}
            rollout_metrics.append(metrics)
        
        rollout_metrics = stack_dict(
            list_dict_extend(
                rollout_metrics
            )
        )
        rollout_metrics = {key: val.mean() for key, val in rollout_metrics.items()}
        self.logger.log(rollout_metrics, prefix="eval/")
        rollout_metrics.update(trajectory_metrics)
        return rollout_metrics
    

class Inference:
    def __init__(
        self, 
        model: ModelBase, 
        env: Env,
        data_miner: SimpleDataMiner,
        checkpoint_path=None,
        seed=None,
        *args,
        **kwargs
    ):
        self.model = instantiate(model)
        self.env = instantiate(env)
        self.data_miner = instantiate(data_miner, model=self.model, env=self.env)

        if checkpoint_path is not None:

            self.checkpointer = CheckPointer(
                self.model,
                None,
                self.env,
                None,
                self.data_miner,
                drop_past=None,
                only_model=True,
            )

            checkpoint_path = checkpoint_path.replace("saved_models", "outputs")
            self.checkpointer.load_checkpoint(checkpoint_path, "best")
        if seed is None:
            self.seed = rand_by_time(0, 10000)
        else:
            self.seed = seed

    def run(self):
        self.data_miner.get_trajectory(seed=self.seed)