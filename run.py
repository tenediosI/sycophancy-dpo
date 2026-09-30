"""Entry point. Examples:

    python run.py stage=split
    python run.py experiment=debug stage=evaluate
    python run.py -m stage=dpo seed=1,2,3
"""
import hydra
from omegaconf import DictConfig

from src.runner import run


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
