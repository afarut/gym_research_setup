import isaacgym
import isaacgymenvs # type: ignore
import torch
import os
import json
import hydra
import subprocess
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from runner import Runner
from common.utils import is_port_running


@hydra.main(version_base=None, config_path='config', config_name='train')
def my_app(cfg: DictConfig):
    if HydraConfig.get().mode.name == "MULTIRUN":
        cfg.name = f"{cfg.name}-seed_{cfg.seed}"

    
    if is_port_running(cfg['streamlit_port']):
        print(f"Streamlit бежит на http://localhost:{cfg['streamlit_port']}")
        print("Если это не streamlit, то измените streamlit_port для запуска")
    else:
        process = subprocess.Popen(
            ["streamlit", "run", "utils/streamlit.py", f"--server.port={cfg['streamlit_port']}", "--server.headless" ,"true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True
        )
    if is_port_running(cfg['tensorboard_port']):
        print(f"Tensorboard бежит на http://localhost:{cfg['tensorboard_port']}")
        print("Если это не tensorboard, то измените tensorboard_port для запуска")

    print(OmegaConf.to_yaml(cfg, resolve=True))
    with open_dict(cfg):
        cfg["train"] = True
    runner = Runner(**cfg)
    runner.train()


if __name__ == "__main__":
    my_app()