"""Experiment runner: dispatches one pipeline stage per call.

Each stage reads its inputs from and writes its outputs to `paths.*`, so stages
can be rerun independently (e.g. re-evaluate without retraining).
"""
import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.data.loaders import load_arc
from src.data.splitter import split_by_question
from src.io_utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def load_split(cfg: DictConfig, name: str) -> list[dict]:
    """Read a saved split, applying data.limit for quick debug runs."""
    records = read_jsonl(Path(cfg.paths.splits) / f"{name}.jsonl")
    return records[: cfg.data.limit] if cfg.data.limit else records


def run_split(cfg: DictConfig) -> None:
    """Step 1: load ARC and write a fixed question-level train/val/test split."""
    records = load_arc(cfg.data.hf_path, list(cfg.data.subsets))
    splits, n_duplicates = split_by_question(
        records, OmegaConf.to_container(cfg.data.fractions), cfg.data.split_seed
    )

    out_dir = Path(cfg.paths.splits)
    for name, split_records in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", split_records)

    manifest = {
        "dataset": cfg.data.name,
        "split_seed": cfg.data.split_seed,
        "n_loaded": len(records),
        "n_duplicates_dropped": n_duplicates,
        "counts": {
            name: {
                "total": len(rs),
                **{s: sum(r["source"] == s for r in rs) for s in cfg.data.subsets},
            }
            for name, rs in splits.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote splits to %s: %s", out_dir, json.dumps(manifest["counts"]))


def not_implemented(step: str):
    def stage(cfg: DictConfig) -> None:
        raise NotImplementedError(f"{step} is not implemented yet.")

    return stage


STAGES = {
    "split": run_split,
    "generate": not_implemented("Step 4 (preference data generation)"),
    "sft": not_implemented("Step 5 (SFT warm-up)"),
    "dpo": not_implemented("Step 6 (DPO training)"),
    "evaluate": not_implemented("Steps 2-3 / 7 (pushback evaluation)"),
    "analyse": not_implemented("Step 7 (statistics across seeds)"),
}


def run(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"Unknown stage {cfg.stage!r}. Choose from {list(STAGES)}.")
    log.info("Running stage %r", cfg.stage)
    STAGES[cfg.stage](cfg)
