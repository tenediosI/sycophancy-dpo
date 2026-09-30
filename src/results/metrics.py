"""Rates with percentile-bootstrap confidence intervals."""
import numpy as np


def bootstrap_rate(outcomes: list[bool], n_bootstrap: int, confidence: float, seed: int = 0) -> dict:
    """Mean of binary outcomes with a percentile-bootstrap CI over items."""
    n = len(outcomes)
    if n == 0:
        return {"rate": None, "ci": [None, None], "n": 0}
    values = np.asarray(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return {"rate": float(values.mean()), "ci": [float(low), float(high)], "n": n}
