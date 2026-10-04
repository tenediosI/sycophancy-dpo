"""Pick the DPO learning rate from a validation sweep with a rule fixed in advance.

Reads artifacts/results/<run>/val/summary.json for the base model and each sweep run.
Rates are averaged over the seen and held-out (test_only) phrasings.

Rules:
- gap (GPU session 2): highest acceptance - capitulation. It rewards the gap between the
  two rates, so it does not stop a run that makes the model refuse valid corrections.
- constrained (from GPU session 3): lowest capitulation among runs whose correction
  acceptance is at most --max-acc-drop below the base model's and whose capitulation is
  at least --min-cap-drop below it (so a run that barely trained cannot win by leaving
  acceptance untouched). If no run qualifies, nothing is chosen.

Usage: python scripts/pick_lr.py --rule constrained --out <name> <base_run> <lr>=<run> ...
Prints the chosen learning rate on the last line ("none" if no run qualifies) and writes
the table to artifacts/results/<dir of base_run>/lr_sweep_<name>.json.
"""
import argparse
import json
from pathlib import Path

RESULTS = Path("artifacts/results")
PHRASINGS = ("seen", "test_only")


def rates(run: str) -> dict:
    summary = json.loads((RESULTS / run / "val" / "summary.json").read_text())
    row = {p: {"capitulation": summary[p]["capitulation_rate"]["rate"],
               "acceptance": summary[p]["correction_acceptance_rate"]["rate"]} for p in PHRASINGS}
    row["capitulation"] = sum(row[p]["capitulation"] for p in PHRASINGS) / len(PHRASINGS)
    row["acceptance"] = sum(row[p]["acceptance"] for p in PHRASINGS) / len(PHRASINGS)
    row["gap"] = row["acceptance"] - row["capitulation"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=["gap", "constrained"], required=True)
    parser.add_argument("--max-acc-drop", type=float, default=0.05)
    parser.add_argument("--min-cap-drop", type=float, default=0.10)
    parser.add_argument("--out", required=True, help="name for lr_sweep_<name>.json")
    parser.add_argument("base_run")
    parser.add_argument("sweep", nargs="+", help="<lr>=<run>")
    args = parser.parse_args()

    sweep = dict(arg.split("=", 1) for arg in args.sweep)
    table = {"base": {"run": args.base_run, **rates(args.base_run)}}
    table |= {lr: {"run": run, **rates(run)} for lr, run in sweep.items()}

    if args.rule == "gap":
        best = max(sweep, key=lambda lr: table[lr]["gap"])
    else:
        base = table["base"]
        eligible = [
            lr for lr in sweep
            if table[lr]["acceptance"] >= base["acceptance"] - args.max_acc_drop
            and table[lr]["capitulation"] <= base["capitulation"] - args.min_cap_drop
        ]
        best = min(eligible, key=lambda lr: table[lr]["capitulation"]) if eligible else "none"

    limits = {"max_acc_drop": args.max_acc_drop, "min_cap_drop": args.min_cap_drop} if args.rule == "constrained" else {}
    out = {"rule": args.rule, **limits, "chosen_lr": best, "table": table}
    path = RESULTS / Path(args.base_run).parent / f"lr_sweep_{args.out}.json"
    path.write_text(json.dumps(out, indent=2))

    for name, row in table.items():
        print(f"{name:>8}  capitulation {row['capitulation']:.3f}  acceptance {row['acceptance']:.3f}  gap {row['gap']:+.3f}")
    if best == "none":
        print(f"No learning rate lowers capitulation by {args.min_cap_drop} while keeping "
              f"acceptance within {args.max_acc_drop} of the base model.")
    print(best)


if __name__ == "__main__":
    main()
