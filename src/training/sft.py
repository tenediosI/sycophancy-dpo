"""Step 5: optional SFT warm-up with LoRA on the chosen replies of the selected pairs.

Data is in TRL's conversational prompt-completion format, so the loss covers only the
assistant's turn-3 reply, not the question, turn 1 or the pushback.
"""
import logging

from datasets import Dataset
from omegaconf import DictConfig

from src.training.common import common_args, init_weights, load_base, load_training_pairs, lora_config, output_dirs, save_run

log = logging.getLogger(__name__)


def to_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list([{"prompt": p["prompt"], "completion": p["chosen"]} for p in pairs])


def run_sft(cfg: DictConfig) -> None:
    from trl import SFTConfig, SFTTrainer

    ckpt_dir, results_dir = output_dirs(cfg)
    train_pairs, train_sel = load_training_pairs(cfg, "train")
    val_pairs, val_sel = load_training_pairs(cfg, "val")

    init = init_weights(cfg)
    model, tokenizer = load_base(init, cfg)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**common_args(cfg, ckpt_dir)),
        train_dataset=to_dataset(train_pairs),
        eval_dataset=to_dataset(val_pairs),
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    save_run(
        trainer, tokenizer, ckpt_dir, results_dir, cfg,
        {"method": "sft", "init": init, "train_pairs": train_sel, "val_pairs": val_sel},
    )
