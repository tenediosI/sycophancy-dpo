"""Load multiple-choice QA datasets into one common record format:

    {"id", "source", "question", "choices": {"A": text, ...}, "answer": "A"}
"""
import logging

from datasets import load_dataset

log = logging.getLogger(__name__)

LETTERS = "ABCDEFGH"


def load_arc(hf_path: str, subsets: list[str]) -> list[dict]:
    """Load every official split of each ARC subset, pooled together."""
    records = []
    for subset in subsets:
        for split in load_dataset(hf_path, subset).values():
            for row in split:
                labels, texts = row["choices"]["label"], row["choices"]["text"]
                # Some ARC items label options 1-4 instead of A-D; normalise to letters.
                letter_of = {label: LETTERS[i] for i, label in enumerate(labels)}
                if row["answerKey"] not in letter_of:
                    log.warning("Skipping %s: answer key not among options", row["id"])
                    continue
                records.append(
                    {
                        "id": row["id"],
                        "source": subset,
                        "question": row["question"],
                        "choices": {letter_of[l]: t for l, t in zip(labels, texts)},
                        "answer": letter_of[row["answerKey"]],
                    }
                )
    return records
