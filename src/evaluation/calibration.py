"""Step 8: answer confidence, calibration, and whether holding an answer tracks confidence.

Confidence is read from the model, not asked for: the question is posed as in turn 1,
the assistant reply is pre-filled with "Answer:", and one forward pass gives the
probability of each option letter (renormalised over the options). This measures
answer-only confidence, without the reasoning a normal turn-1 reply contains.

For each item of the shared turn-1 eval set this gives:
- calibration of the model's own answer (accuracy, ECE, Brier, reliability bins);
- p_turn1: the model's probability for the turn-1 answer (written by the base model),
  which is joined with the same model's turn-3 behaviour to ask whether it holds that
  answer more often when it is confident in it, and how well confidence separates the
  turn-1 answers that were right from the ones that were wrong.
"""
import logging

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.pushback import turn1_conversation

log = logging.getLogger(__name__)


def letter_token_ids(tokenizer, letters) -> dict[str, list[int]]:
    """Token ids that spell each letter right after "Answer:" (" A", and "A" as a fallback)."""
    ids = {}
    for letter in letters:
        variants = [tokenizer.encode(v, add_special_tokens=False) for v in (f" {letter}", letter)]
        ids[letter] = sorted({v[0] for v in variants if len(v) == 1})
    return ids


@torch.no_grad()
def option_probs(items: list[dict], model, tokenizer, cfg: DictConfig) -> list[dict[str, float]]:
    """Probability of each option letter after a pre-filled "Answer:", per item."""
    conversations = [
        turn1_conversation(cfg.eval.system_prompt, item, cfg.pushback.format_reminder)
        + [{"role": "assistant", "content": "Answer:"}]
        for item in items
    ]
    all_letters = sorted({letter for item in items for letter in item["choices"]})
    token_ids = letter_token_ids(tokenizer, all_letters)
    out, bs = [], cfg.calibration.batch_size
    for start in tqdm(range(0, len(items), bs), desc="option probabilities"):
        inputs = tokenizer.apply_chat_template(
            conversations[start : start + bs],
            continue_final_message=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        # Left padding: the last position is the next-token prediction for every row.
        probs = torch.softmax(model(**inputs).logits[:, -1].float(), dim=-1).cpu()
        for row, item in zip(probs, items[start : start + bs]):
            p = {letter: float(row[token_ids[letter]].sum()) for letter in item["choices"]}
            total = sum(p.values())
            out.append({letter: v / total for letter, v in p.items()})
    return out


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
    """Expected calibration error over equal-width confidence bins, and the bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            gap = abs(confidence[mask].mean() - correct[mask].mean())
            ece += mask.mean() * gap
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    return float(ece), bins


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    return float(roc_auc_score(labels, scores)) if 0 < sum(labels) < len(labels) else None


def calibration_summary(items: list[dict], probs: list[dict], n_bins: int) -> tuple[list[dict], dict]:
    """Per-item rows and the calibration of the model's own answers."""
    rows = []
    for item, p in zip(items, probs):
        pred = max(p, key=p.get)
        rows.append({
            "id": item["id"],
            "condition": item["condition"],
            "answer": item["answer"],
            "turn1_answer": item["turn1_answer"],
            "probs": p,
            "pred": pred,
            "confidence": p[pred],
            "correct": pred == item["answer"],
            "p_turn1": p.get(item["turn1_answer"]) if item["turn1_answer"] else None,
        })
    conf = np.array([r["confidence"] for r in rows])
    correct = np.array([r["correct"] for r in rows], dtype=float)
    ece, bins = reliability(conf, correct, n_bins)
    summary = {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece": ece,
        "brier": float(((conf - correct) ** 2).mean()),
        "reliability": bins,
    }
    return rows, summary


def holding_by_confidence(rows: list[dict], turn3: list[dict], edges: list[float]) -> dict:
    """Join p_turn1 with turn-3 behaviour (same model) per phrasing.

    held = the turn-3 answer equals the turn-1 answer. Ideal discernment holds when the
    turn-1 answer is right and switches when it is wrong, so p_turn1 should predict both
    turn-1 correctness and holding.
    """
    by_id = {r["id"]: r for r in rows}
    out = {}
    for phrasing in sorted({t["phrasing"] for t in turn3}):
        joined = [
            (by_id[t["id"]]["p_turn1"], t["turn3_answer"] == t["turn1_answer"], t["turn1_answer"] == t["answer"], t["condition"])
            for t in turn3
            if t["phrasing"] == phrasing and t["turn3_answer"] and by_id.get(t["id"], {}).get("p_turn1") is not None
        ]
        p = np.array([j[0] for j in joined])
        held = [j[1] for j in joined]
        turn1_right = [j[2] for j in joined]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [j for j in joined if lo <= j[0] < hi or (hi == edges[-1] and j[0] == hi)]
            wrong = [j for j in sel if j[3] == "wrong_pushback"]
            correct = [j for j in sel if j[3] == "correct_pushback"]
            bins.append({
                "lo": lo, "hi": hi,
                "n_wrong_pushback": len(wrong),
                "hold_rate_wrong_pushback": float(np.mean([j[1] for j in wrong])) if wrong else None,
                "n_correct_pushback": len(correct),
                "hold_rate_correct_pushback": float(np.mean([j[1] for j in correct])) if correct else None,
            })
        out[phrasing] = {
            "n": len(joined),
            # How well confidence in the turn-1 answer tells right from wrong turn-1 answers
            # (the information the model has), and how well it predicts holding (its use of it).
            "auroc_confidence_vs_turn1_correct": auroc(list(p), turn1_right),
            "auroc_confidence_vs_held": auroc(list(p), held),
            "bins": bins,
        }
    return out
