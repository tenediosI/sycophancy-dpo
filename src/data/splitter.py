"""Question-level train/val/test split: no question appears in more than one split."""
import random
import re
from collections import defaultdict


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_by_question(
    records: list[dict], fractions: dict[str, float], seed: int, stratify_key: str = "source"
) -> tuple[dict[str, list[dict]], int]:
    """Deduplicate by question text, then split each stratum by the given fractions.

    Returns the splits and the number of duplicate questions dropped.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")

    seen, unique = set(), []
    for record in records:
        key = _normalise(record["question"])
        if key not in seen:
            seen.add(key)
            unique.append(record)

    strata = defaultdict(list)
    for record in unique:
        strata[record[stratify_key]].append(record)

    rng = random.Random(seed)
    splits = {name: [] for name in fractions}
    for stratum in sorted(strata):
        items = sorted(strata[stratum], key=lambda r: r["id"])  # order-independent of loading
        rng.shuffle(items)
        start = 0
        for i, (name, fraction) in enumerate(fractions.items()):
            # The last split takes the remainder so no item is lost to rounding.
            end = len(items) if i == len(fractions) - 1 else start + round(fraction * len(items))
            splits[name].extend(items[start:end])
            start = end

    return splits, len(records) - len(unique)
