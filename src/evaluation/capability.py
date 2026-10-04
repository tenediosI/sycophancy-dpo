"""Capability benchmarks with lm-eval (MMLU, GSM8K, IFEval), before and after training.

Each task runs on a fixed subset (lm-eval's `limit` takes the first N documents, or the
first N per subject for MMLU), so every model is scored on the same items. Per-item
scores are saved alongside the aggregates, so base and trained models can be compared
with paired tests (Step 7).
"""
import logging

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Fields of lm-eval's logged samples kept per item: enough to pair items across models
# and to read a response by hand. Prompts and few-shot examples are dropped (large).
SAMPLE_FIELDS = ("doc_id", "doc_hash", "filter", "target", "resps", "filtered_resps")


def load_lm(cfg: DictConfig):
    """Wrap the configured model (plus an optional LoRA adapter) for lm-eval."""
    from lm_eval.models.huggingface import HFLM

    from src.models.factory import weights_path

    device = "cpu" if cfg.model.device_map == "cpu" else "cuda"
    return HFLM(
        pretrained=weights_path(cfg.model),
        peft=cfg.capability.adapter,
        dtype=cfg.model.dtype,
        device=device,
        batch_size=cfg.capability.batch_size,
    )


def run_capability(lm, cfg: DictConfig) -> tuple[dict, list[dict]]:
    """Run every configured task on its fixed subset. Returns (aggregates, per-item rows)."""
    from lm_eval import simple_evaluate

    cap = cfg.capability
    gen_kwargs = OmegaConf.to_container(cap.gen_kwargs) if cap.gen_kwargs else None
    aggregates, rows = {}, []
    for task, limit in cap.tasks.items():
        log.info("Running %s (limit=%s)", task, limit)
        # One call per task because the subset size differs per task.
        out = simple_evaluate(
            model=lm,
            tasks=[task],
            limit=limit,
            apply_chat_template=cap.apply_chat_template,
            fewshot_as_multiturn=cap.apply_chat_template,
            gen_kwargs=gen_kwargs,
            bootstrap_iters=cap.bootstrap_iters,
            log_samples=True,
            random_seed=cfg.seed,
        )
        # MMLU is a group: keep the group score and the per-subject scores.
        for name, metrics in out["results"].items():
            aggregates[name] = {k: v for k, v in metrics.items() if k != "alias" and v != "N/A"}
        for subtask, samples in out["samples"].items():
            for s in samples:
                scores = {m: s[m] for m in s["metrics"]}
                rows.append({"task": subtask, **{f: s[f] for f in SAMPLE_FIELDS}, **scores})
    return aggregates, rows


def headline(aggregates: dict) -> dict:
    """The one number per benchmark reported in the write-up.

    GSM8K uses flexible-extract (last number in the reply), not strict-match (the
    few-shot "The answer is N" format): training on "Answer: X" replies may shift the
    output format, and that should not count as lost arithmetic. Both are in the summary.
    """
    picks = {
        "mmlu": "acc,none",
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_strict_acc,none",
    }
    out = {}
    for task, metric in picks.items():
        if task in aggregates and metric in aggregates[task]:
            out[task] = {
                "metric": metric,
                "value": aggregates[task][metric],
                "stderr": aggregates[task].get(metric.replace(",", "_stderr,", 1)),
            }
    return out
