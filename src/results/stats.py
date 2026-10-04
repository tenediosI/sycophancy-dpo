"""Step 7: paired comparisons of trained models against the base model, and across seeds.

Every model answers the same items (the shared turn-1 eval set and the same benchmark
subsets), so each comparison is paired by item:
- McNemar's exact test on the items where the two models disagree;
- a paired bootstrap CI for the difference in rates (items resampled, same for both);
- Holm's correction over every test in the analysis.
Discernment = correction acceptance - capitulation (0 = the model reacts the same whether
the user is right or wrong); its CI resamples the two conditions' items separately.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

# Headline item-level score per benchmark: (metric field, lm-eval filter).
CAPABILITY_METRICS = {
    "mmlu": ("acc", "none"),
    "gsm8k": ("exact_match", "flexible-extract"),
    "ifeval": ("prompt_level_strict_acc", "none"),
}
PHRASINGS = ("seen", "test_only")


def pushback_outcomes(rows: list[dict]) -> dict[str, dict]:
    """{metric: {(id, phrasing): outcome}} with outcome True = the failure / success the
    metric counts. Unparseable turn-3 answers are left out, as in the headline rates."""
    out = defaultdict(dict)
    for r in rows:
        if r["turn3_answer"] is None:
            continue
        key = (r["id"], r["phrasing"])
        if r["condition"] == "wrong_pushback":
            out[f"capitulation/{r['phrasing']}"][key] = not r["correct_behaviour"]
        elif r["condition"] == "correct_pushback":
            out[f"acceptance/{r['phrasing']}"][key] = bool(r["correct_behaviour"])
    return out


def capability_outcomes(rows: list[dict]) -> dict[str, dict]:
    """{benchmark: {(task, doc_id): correct}} for each benchmark's headline metric."""
    out = defaultdict(dict)
    for r in rows:
        bench = "mmlu" if r["task"].startswith("mmlu") else r["task"]
        if bench not in CAPABILITY_METRICS:
            continue
        metric, filt = CAPABILITY_METRICS[bench]
        if r["filter"] == filt and metric in r:
            out[bench][(r["task"], r["doc_id"])] = bool(r[metric])
    return out


def paired(base: dict, model: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(base.keys() & model.keys())
    return (np.array([base[k] for k in keys], dtype=float), np.array([model[k] for k in keys], dtype=float))


def mcnemar_exact(base: np.ndarray, model: np.ndarray) -> dict:
    b = int(((base == 1) & (model == 0)).sum())  # only the base model scores
    c = int(((base == 0) & (model == 1)).sum())  # only the trained model scores
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"base_only": b, "model_only": c, "p": float(p)}


def paired_bootstrap_diff(base: np.ndarray, model: np.ndarray, n_bootstrap: int, confidence: float, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=(n_bootstrap, len(base)))
    diffs = model[idx].mean(axis=1) - base[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return [float(x) for x in np.quantile(diffs, [alpha, 1 - alpha])]


def compare(base: dict, model: dict, n_bootstrap: int, confidence: float) -> dict:
    """Paired comparison of one metric between the base and a trained model."""
    b, m = paired(base, model)
    return {
        "n": len(b),
        "base": float(b.mean()),
        "model": float(m.mean()),
        "diff": float(m.mean() - b.mean()),
        "diff_ci": paired_bootstrap_diff(b, m, n_bootstrap, confidence),
        **mcnemar_exact(b, m),
    }


def discernment(base: dict[str, dict], model: dict[str, dict], phrasing: str, n_bootstrap: int, confidence: float, seed: int = 0) -> dict:
    """Acceptance - capitulation for both models, and a paired bootstrap CI of the change."""
    acc_b, acc_m = paired(base[f"acceptance/{phrasing}"], model[f"acceptance/{phrasing}"])
    cap_b, cap_m = paired(base[f"capitulation/{phrasing}"], model[f"capitulation/{phrasing}"])
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(acc_b), size=(n_bootstrap, len(acc_b)))
    ic = rng.integers(0, len(cap_b), size=(n_bootstrap, len(cap_b)))
    change = (acc_m[ia].mean(1) - cap_m[ic].mean(1)) - (acc_b[ia].mean(1) - cap_b[ic].mean(1))
    alpha = (1 - confidence) / 2
    j_base, j_model = acc_b.mean() - cap_b.mean(), acc_m.mean() - cap_m.mean()
    return {
        "base": float(j_base),
        "model": float(j_model),
        "diff": float(j_model - j_base),
        "diff_ci": [float(x) for x in np.quantile(change, [alpha, 1 - alpha])],
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (family-wise error control), same keys as the input."""
    order = sorted(pvalues, key=pvalues.get)
    m, adjusted, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def report(summary: dict, base_name: str) -> str:
    """Markdown tables: each run against the base model, then the mean over seeds."""
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    s = summary["settings"]
    runs = summary["runs"]
    lines = [
        f"# Analysis: {s['name']}",
        "",
        f"Each run is compared with `{base_name}` on the same items ({s['split']} split for pushback). "
        f"Differences are run minus base, in points, with paired bootstrap {int(100 * s['confidence'])}% CIs. "
        f"p = McNemar exact; p_holm = Holm-corrected over all {sum(len(r['metrics']) for r in runs.values())} tests; "
        f"* = significant at {s['alpha']} after correction. "
        "base-only / run-only = items only that model gets 'right' for the metric "
        "(for capitulation, 'right' means it capitulated).",
        "",
    ]
    for name in next(iter(runs.values()))["metrics"]:
        lines += [f"## {name}", "", "| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |", "|---|---|---|---|---|---|---|---|"]
        for run, r in runs.items():
            m = r["metrics"][name]
            lines.append(
                f"| {run} | {m['n']} | {pct(m['base'])} | {pct(m['model'])} | "
                f"{pct(m['diff'])} [{pct(m['diff_ci'][0])}, {pct(m['diff_ci'][1])}] | "
                f"{m['base_only']} / {m['model_only']} | {m['p']:.2g} | {m['p_holm']:.2g}{' *' if m['significant'] else ''} |"
            )
        lines.append("")
    lines += ["## Discernment (acceptance - capitulation)", "", "| run | phrasing | base | run | diff [CI] |", "|---|---|---|---|---|"]
    for run, r in runs.items():
        for p, d in r["discernment"].items():
            lines.append(
                f"| {run} | {p} | {pct(d['base'])} | {pct(d['model'])} | {pct(d['diff'])} [{pct(d['diff_ci'][0])}, {pct(d['diff_ci'][1])}] |"
            )
    lines += ["", "## Across seeds (mean, SD, range)", "", "| metric | run mean (SD) | diff mean (SD) | diff range |", "|---|---|---|---|"]
    for name, a in summary["across_seeds"].items():
        sd = lambda d: pct(d["sd"]) if d["sd"] is not None else "-"  # noqa: E731
        lines.append(
            f"| {name} | {pct(a['model']['mean'])} ({sd(a['model'])}) | {pct(a['diff']['mean'])} ({sd(a['diff'])}) | "
            f"{pct(a['diff']['min'])} to {pct(a['diff']['max'])} |"
        )
    return "\n".join(lines) + "\n"


def across_seeds(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
