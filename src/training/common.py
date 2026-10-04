"""Shared pieces of SFT and DPO training: data, LoRA, arguments and saving.

Each run writes the LoRA adapter and a merged copy of the model to
`artifacts/checkpoints/<run_name>/` (not committed), and its training log, pair selection
and config to `artifacts/results/<run_name>/train/` (committed). The merged model is what
evaluation loads (`model.checkpoint=.../merged`) and what DPO starts from after SFT.
"""
import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.preferences import CONDITIONS, select_pairs
from src.io_utils import prepare_output_dir, read_jsonl

log = logging.getLogger(__name__)


def load_training_pairs(cfg: DictConfig, split: str) -> tuple[list[dict], dict]:
    """Read the pairs saved by `generate` and select the ones to train or validate on."""
    p = cfg.train.pairs
    path = Path(cfg.paths.preferences) / p.run_name / split / "pairs.jsonl"
    pairs, summary = select_pairs(read_jsonl(path), p.wrong_per_correct, p.prefer_sampled_chosen, cfg.seed)
    if cfg.train.max_pairs:  # debug runs: pairs are shuffled, so both conditions stay represented
        pairs = pairs[: cfg.train.max_pairs]
    # Without both conditions the model can lower capitulation by never changing its answer.
    missing = [c for c in CONDITIONS if not any(pair["condition"] == c for pair in pairs)]
    if missing:
        raise ValueError(f"No {missing} pairs selected from {path}; refusing to train on one condition only.")
    log.info("Selected %d %s pairs from %s: %s", len(pairs), split, path, json.dumps(summary))
    return pairs, summary | {"source": str(path), "n_used": len(pairs)}


def lora_config(cfg: DictConfig):
    from peft import LoraConfig

    lora = cfg.model.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        task_type="CAUSAL_LM",
    )


def init_weights(cfg: DictConfig) -> str:
    """Weights training starts from: the base model, or the merged SFT model for DPO after SFT."""
    if cfg.train.get("init_from", "base") == "sft":
        path = Path(cfg.paths.checkpoints) / cfg.train.sft_run_name / "merged"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run stage=sft train=sft first.")
        return str(path)
    return cfg.model.checkpoint or cfg.model.name


def output_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """(checkpoint dir, results dir), both checked before any model is loaded."""
    ckpt = prepare_output_dir(Path(cfg.paths.checkpoints) / cfg.train.run_name, cfg.overwrite)
    results = prepare_output_dir(Path(cfg.paths.results) / cfg.train.run_name / "train", cfg.overwrite)
    return ckpt, results


def common_args(cfg: DictConfig, ckpt_dir: Path) -> dict:
    """TrainingArguments shared by SFTConfig and DPOConfig."""
    t = cfg.train
    return dict(
        output_dir=str(ckpt_dir / "trainer"),
        seed=cfg.seed,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        max_length=t.max_length,
        bf16=cfg.model.dtype == "bfloat16",
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="no",  # the final adapter and merged model are saved by save_run
        report_to="none",
        use_cpu=cfg.model.device_map == "cpu",
    )


def load_base(path: str, cfg: DictConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, cfg.model.dtype))
    return model, tokenizer


def save_run(trainer, tokenizer, ckpt_dir: Path, results_dir: Path, cfg: DictConfig, summary: dict) -> None:
    """Save the adapter, a merged model, the training log and the run's config."""
    trainer.model.save_pretrained(ckpt_dir / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(ckpt_dir / "merged")
    tokenizer.save_pretrained(ckpt_dir / "merged")

    history = trainer.state.log_history
    evals = [h for h in history if "eval_loss" in h]
    summary = summary | {
        "final_train": history[-1] if history else {},
        "final_eval": evals[-1] if evals else {},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (results_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    log.info("Saved %s and %s", ckpt_dir, results_dir)
