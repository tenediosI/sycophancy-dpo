"""Upload a trained LoRA adapter to the Hugging Face Hub as a private model repo.

Usage (needs HF_TOKEN in the environment, a token with write access):
    python scripts/push_adapter.py artifacts/checkpoints/<run>/adapter <user>/<repo> <run>

The repo is created private; make it public on the Hub when you are ready. A short model
card is written next to the adapter; the full card is part of the write-up (Step 10).
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
tags: [dpo, lora, sycophancy]
---

# Sycophancy-resistant LoRA adapter for Qwen2.5-1.5B-Instruct

LoRA adapter trained with DPO to hold a correct answer under wrong user pushback while
still accepting valid corrections. Training run: `{run}`.

Code, data, evaluation and results: https://github.com/tenediosI/sycophancy-dpo

Training settings: {settings}
"""


def main() -> None:
    adapter_dir, repo_id, run = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    summary_path = Path("artifacts/results") / run / "train" / "summary.json"
    settings = {}
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        settings = {"method": s.get("method"), "beta": s.get("beta"), "pairs": s.get("train_pairs", {}).get("selected")}
    (adapter_dir / "README.md").write_text(CARD.format(run=run, settings=json.dumps(settings)), encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, commit_message=f"Upload adapter from {run}")
    print(f"Uploaded {adapter_dir} to https://huggingface.co/{repo_id} (private)")


if __name__ == "__main__":
    main()
