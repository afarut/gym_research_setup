import os
import torch
import shutil
from model.base import AutoModel
from gymnasium import Env
from logger.base import LoggerBase
from pathlib import Path
from core.dataminer import SimpleDataMiner
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


class CheckPointer:
    def __init__(
        self,
        model: AutoModel,
        optimizer: torch.optim.Optimizer,
        env: Env,
        logger: LoggerBase,
        data_miner: SimpleDataMiner,
        target_metric="rewards",
        best_metric=float("-inf"),
        drop_past=False,
        only_model=False,
        scheduler=None,
    ):
        self.model = model
        self.env = env
        self.only_model = only_model
        
        if not only_model:
            self.logger = logger
            self.optimizer = optimizer
            self.target_metric = target_metric
            self.best_metric = best_metric
            self.data_miner = data_miner
            self.scheduler = scheduler

            _, self.path = self.logger.log_dir.split("outputs")
            self.path = "saved_models" + self.path

            self.drop_past = drop_past
            os.makedirs(self.path + "/last")
            os.makedirs(self.path + "/best")


    def load_checkpoint(self, log_dir, prefix):
        _, path = log_dir.split("outputs")
        path = "saved_models" + path

        checkpoint = torch.load(path + f"/{prefix}/model.pth", map_location=self.model.device)
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.only_model:
            checkpoint = torch.load(path + f"/{prefix}/optimizer.pth", map_location=self.model.device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler is not None:
                checkpoint = torch.load(path + f"/{prefix}/scheduler.pth", map_location=self.model.device)
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            checkpoint = torch.load(path + f"/{prefix}/meta.pth")
            cfg = OmegaConf.load(f"{log_dir}/.hydra/config.yaml")
            assert cfg["env"]["id"] == self.env.spec.id

            # Continue steps TensorBoard
            self.data_miner.seed = checkpoint["data_miner_seed"]
            for key, value in checkpoint["logger_steps"].items():
                self.logger.steps[key] = value

            # Copy TensorBoard dir
            if not self.drop_past:
                os.makedirs(self.logger.log_dir + "/tensorboard", exist_ok=True)

                ea = event_accumulator.EventAccumulator(log_dir + "/tensorboard")
                ea.Reload()

                for tag in ea.Tags().get("scalars", []):
                    events = ea.Scalars(tag)
                    for e in events:
                        if e.step < self.logger.steps[key]:
                            self.logger.writer.add_scalar(tag, e.value, e.step)
            
            self.best_metric = checkpoint["best_metric"]

            if self.drop_past:
                return 0
            return checkpoint["epoch"]


    def save(self, epoch, metric):
        self.save_checkpoint(epoch, "last")
        if metric[self.target_metric] <= self.best_metric:
            return

        self.save_checkpoint(epoch, "best")
        self.best_metric = metric[self.target_metric]

    def save_checkpoint(self, epoch, prefix):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, self.path + f"/{prefix}/model.pth")

        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.path + f"/{prefix}/optimizer.pth")

        if self.scheduler is not None:
            checkpoint = {
                "scheduler_state_dict": self.scheduler.state_dict(),
            }
            torch.save(checkpoint, self.path + f"/{prefix}/scheduler.pth")

        checkpoint = {
            "data_miner_seed": self.data_miner.seed,
            "logger_steps": dict(self.logger.steps),
            "best_metric": float(self.best_metric),
            "epoch": epoch,
        }
        torch.save(checkpoint, self.path + f"/{prefix}/meta.pth")







# saved_models/2025-10-17/14-10-21-0//last/model.pth
# body.0.weight tensor(3.6804)
# body.0.bias tensor(1.6123)
# body.2.weight tensor(5.6973)
# body.2.bias tensor(0.2829)
# scale_head.weight tensor(0.8370)
# scale_head.bias tensor(0.0346)
# std_head.weight tensor(0.7005)
# std_head.bias tensor(0.0897)


# saved_models/2025-10-17/14-10-21-0//last/model.pth
# body.0.weight tensor(3.6804)
# body.0.bias tensor(1.6123)
# body.2.weight tensor(5.6973)
# body.2.bias tensor(0.2829)
# scale_head.weight tensor(0.8370)
# scale_head.bias tensor(0.0346)
# std_head.weight tensor(0.7005)
# std_head.bias tensor(0.0897)
