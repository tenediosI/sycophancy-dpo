# Sycophancy-resistant preference tuning

Fine-tune a small open-weight chat model with SFT + DPO (LoRA) so that it holds a
correct answer when a user pushes back wrongly, and accepts a correction when the
user is right, while preserving general capability.

Work in progress. See `../dpo-sycophancy-project-guide.md` for the full plan.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

## Pipeline

Each stage reads from and writes to `artifacts/`, so stages can be rerun independently.
Configuration lives in `conf/` (Hydra); every run saves its full config under `outputs/`.

| Stage | Command | Status |
|---|---|---|
| 1. Split ARC by question | `python run.py stage=split` | done |
| 4. Generate preference pairs | `python run.py stage=generate` | todo |
| 5. SFT warm-up | `python run.py stage=sft train=sft` | todo |
| 6. DPO | `python run.py stage=dpo` | todo |
| 2-3, 7. Pushback evaluation | `python run.py stage=evaluate` | todo |
| 7. Statistics across seeds | `python run.py stage=analyse` | todo |

Add `experiment=debug` to any command for a tiny CPU run with a 0.5B model and
20 questions per split, to check the code on a laptop. Run several seeds with
`python run.py -m stage=dpo seed=1,2,3`.

## Data

ARC (Easy + Challenge), all official splits pooled, deduplicated by question text
(35 duplicates dropped) and re-split 70/10/20 by question, stratified by subset,
with `data.split_seed=42`:

| Split | Total | Easy | Challenge |
|---|---|---|---|
| train | 5427 | 3625 | 1802 |
| val | 775 | 518 | 257 |
| test | 1550 | 1035 | 515 |
