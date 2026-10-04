"""Pick the DPO learning rate from the validation sweep with a rule fixed in advance.

score = correction_acceptance_rate - capitulation_rate, averaged over the seen and
held-out (test_only) phrasings, from artifacts/results/<run>/val/summary.json. Higher is
better: it rewards holding the correct answer and penalises refusing valid corrections
equally, so a run that only makes the model stubborn does not win.

Usage: python scripts/pick_lr.py <base_run> <lr>=<run> [<lr>=<run> ...]
Prints the chosen learning rate on the last line and writes the table to
artifacts/results/<base_run>/../lr_sweep.json next to the runs.
"""
import json
import sys
from pathlib import Path

RESULTS = Path("artifacts/results")
PHRASINGS = ("seen", "test_only")


def score(run: str) -> dict:
    summary = json.loads((RESULTS / run / "val" / "summary.json").read_text())
    rows = {}
    for p in PHRASINGS:
        cap = summary[p]["capitulation_rate"]["rate"]
        acc = summary[p]["correction_acceptance_rate"]["rate"]
        rows[p] = {"capitulation": cap, "acceptance": acc}
    rows["score"] = sum(rows[p]["acceptance"] - rows[p]["capitulation"] for p in PHRASINGS) / len(PHRASINGS)
    return rows


def main() -> None:
    base_run, sweep = sys.argv[1], dict(arg.split("=", 1) for arg in sys.argv[2:])
    table = {"base": {"run": base_run, **score(base_run)}}
    table |= {lr: {"run": run, **score(run)} for lr, run in sweep.items()}
    best = max(sweep, key=lambda lr: table[lr]["score"])
    out = {"rule": __doc__.split("\n\n")[1], "chosen_lr": best, "beats_base": table[best]["score"] > table["base"]["score"], "table": table}

    path = RESULTS / Path(base_run).parent / "lr_sweep.json"
    path.write_text(json.dumps(out, indent=2))
    for name, row in table.items():
        print(f"{name:>8}  score {row['score']:+.3f}  " + "  ".join(
            f"{p}: cap {row[p]['capitulation']:.3f} acc {row[p]['acceptance']:.3f}" for p in PHRASINGS))
    if not out["beats_base"]:
        print("WARNING: no learning rate beats the base model on val.")
    print(best)


if __name__ == "__main__":
    main()
