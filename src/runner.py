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
from src.io_utils import prepare_output_dir, read_jsonl, write_jsonl

# Model-dependent imports (torch, transformers) happen inside the stages that need
# them, so `stage=split` works without installing the ML stack.

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


def run_evaluate(cfg: DictConfig) -> None:
    """Steps 2-3 (and later 7): pushback evaluation with bootstrap CIs."""
    from src.evaluation.pushback_eval import build_eval_set, run_pushback, spot_check_report, summarise
    from src.models.factory import load_model

    out_dir = prepare_output_dir(Path(cfg.paths.results) / cfg.eval.run_name / cfg.eval.split, cfg.overwrite)
    records = load_split(cfg, cfg.eval.split)
    model, tokenizer = load_model(cfg.model)

    # Turn 1 comes from the base model and is shared by every model evaluated later.
    model_slug = cfg.model.name.split("/")[-1]
    eval_set_path = Path(cfg.paths.eval_sets) / f"{cfg.eval.split}_{model_slug}_n{len(records)}.jsonl"
    if eval_set_path.exists():
        items = read_jsonl(eval_set_path)
        log.info("Reusing eval set %s", eval_set_path)
    else:
        items = build_eval_set(records, model, tokenizer, cfg)
        write_jsonl(eval_set_path, items)
        log.info("Wrote eval set %s", eval_set_path)

    rows = run_pushback(items, model, tokenizer, cfg)
    summary = {"model": cfg.model.name, "split": cfg.eval.split, "eval_set": str(eval_set_path)}
    summary |= summarise(items, rows, cfg)

    write_jsonl(out_dir / "turn3.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "spot_check.md").write_text(
        spot_check_report(items, rows, cfg.eval.spot_check_n, cfg.eval.eval_set_seed), encoding="utf-8"
    )
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    log.info("Results in %s:\n%s", out_dir, json.dumps(summary, indent=2))


def run_generate(cfg: DictConfig) -> None:
    """Step 4: preference pairs by rejection sampling from the base model."""
    from transformers import set_seed

    from src.data.preferences import build_preferences, spot_check_report
    from src.models.factory import load_model

    out_dir = prepare_output_dir(
        Path(cfg.paths.preferences) / cfg.generate.run_name / cfg.generate.split, cfg.overwrite
    )
    records = load_split(cfg, cfg.generate.split)
    set_seed(cfg.seed)
    model, tokenizer = load_model(cfg.model)

    pairs, samples, summary = build_preferences(records, model, tokenizer, cfg)
    summary = {"model": cfg.model.name, "split": cfg.generate.split} | summary

    write_jsonl(out_dir / "pairs.jsonl", pairs)
    write_jsonl(out_dir / "samples.jsonl", samples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "spot_check.md").write_text(
        spot_check_report(pairs, cfg.generate.spot_check_n, cfg.seed), encoding="utf-8"
    )
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    log.info("Wrote %d pairs to %s:\n%s", len(pairs), out_dir, json.dumps(summary, indent=2))


def not_implemented(step: str):
    def stage(cfg: DictConfig) -> None:
        raise NotImplementedError(f"{step} is not implemented yet.")

    return stage


STAGES = {
    "split": run_split,
    "generate": run_generate,
    "sft": not_implemented("Step 5 (SFT warm-up)"),
    "dpo": not_implemented("Step 6 (DPO training)"),
    "evaluate": run_evaluate,
    "analyse": not_implemented("Step 7 (statistics across seeds)"),
}


def run(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"Unknown stage {cfg.stage!r}. Choose from {list(STAGES)}.")
    log.info("Running stage %r", cfg.stage)
    STAGES[cfg.stage](cfg)
