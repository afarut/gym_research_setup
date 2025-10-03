import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from omegaconf import open_dict
from runner import Inference


@hydra.main(version_base=None, config_path='config', config_name='inference')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    runner = Inference(**cfg)
    runner.run()


if __name__ == "__main__":
    my_app()