"""Step 6: DPO with LoRA, from the base model or from the merged SFT model.

With a LoRA adapter and no ref_model, TRL uses the same model with the adapter disabled
as the reference, i.e. the starting point (base or SFT). Only prompt/chosen/rejected are
passed to the trainer; the pairs' metadata columns are dropped.
"""
import logging

from datasets import Dataset
from omegaconf import DictConfig

from src.training.common import common_args, init_weights, load_base, load_training_pairs, lora_config, output_dirs, save_run

log = logging.getLogger(__name__)


def to_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    )


def run_dpo(cfg: DictConfig) -> None:
    from trl import DPOConfig, DPOTrainer

    ckpt_dir, results_dir = output_dirs(cfg)
    train_pairs, train_sel = load_training_pairs(cfg, "train")
    val_pairs, val_sel = load_training_pairs(cfg, "val")

    init = init_weights(cfg)
    model, tokenizer = load_base(init, cfg)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**common_args(cfg, ckpt_dir), beta=cfg.train.beta),
        train_dataset=to_dataset(train_pairs),
        eval_dataset=to_dataset(val_pairs),
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    save_run(
        trainer, tokenizer, ckpt_dir, results_dir, cfg,
        {"method": "dpo", "init": init, "beta": cfg.train.beta, "train_pairs": train_sel, "val_pairs": val_sel},
    )
