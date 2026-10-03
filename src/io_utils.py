import json
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    """Create a stage's output directory, refusing to reuse one that already holds files.

    Called before any model is loaded, so a clash fails immediately instead of after a
    long GPU run.
    """
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{path} already contains results. Choose a different run_name, "
            "or pass overwrite=true to replace them."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
