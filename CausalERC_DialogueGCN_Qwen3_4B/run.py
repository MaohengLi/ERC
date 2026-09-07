"""One-command training followed by evaluation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "outputs" / "qwen3_4b"
    output.mkdir(parents=True, exist_ok=True)
    train = subprocess.run([sys.executable, "-u", "train.py", "--config", args.config], cwd=root)
    if train.returncode != 0:
        return train.returncode
    evaluate = subprocess.run([sys.executable, "-u", "evaluate.py", "--config", args.config], cwd=root)
    return evaluate.returncode

if __name__ == "__main__":
    raise SystemExit(main())
