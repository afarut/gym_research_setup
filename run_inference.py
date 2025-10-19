import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from omegaconf import open_dict
from runner import Inference


@hydra.main(version_base=None, config_path='config', config_name='inference')
def my_app(cfg: DictConfig):
    with open_dict(cfg):
        if "headless" in cfg["env"]:
            cfg["env"]["headless"] = False
            cfg["env"]["force_render"] = True
            del cfg["env"]["render_mode"]
            del cfg["env"]["vectorization_mode"]
    print(OmegaConf.to_yaml(cfg, resolve=True))
    runner = Inference(**cfg)
    runner.run()


if __name__ == "__main__":
    my_app()