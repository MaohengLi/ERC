"""Run one full-data seed, then evaluate its dev-selected checkpoint on test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


PACKAGE = "causal_dialoguegcn_mllm_paper_aligned_4b"
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--llm_micro_batch", type=int, default=2)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "experiment_state.json"
    if state_path.exists():
        raise RuntimeError("This run directory is already in use; choose a new directory")
    state = {"status": "preparing", "started_at": now(), "runner_pid": os.getpid(),
             "run_dir": str(run_dir), "epochs_requested": args.epochs, "seed": 42,
             "full_data": True}

    def write_state(**values):
        state.update(values)
        state["updated_at"] = now()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(state_path)

    write_state()
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for source in sorted((ROOT / PACKAGE).glob("*.py")):
        shutil.copy2(source, snapshot / source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    train_command = [sys.executable, "-u", "-m", f"{PACKAGE}.train",
                     "--output_dir", str(run_dir), "--epochs", str(args.epochs),
                     "--seed", "42", "--max_length", str(args.max_length),
                     "--llm_micro_batch", str(args.llm_micro_batch),
                     "--log_every", "1", "--max_train_dialogues", "0",
                     "--max_dev_dialogues", "0"]
    eval_command = [sys.executable, "-u", "-m", f"{PACKAGE}.evaluate",
                    "--checkpoint_dir", str(run_dir / "best_model"), "--split", "test",
                    "--llm_micro_batch", str(args.llm_micro_batch),
                    "--output_json", str(run_dir / "test_metrics.json")]
    (run_dir / "manifest.json").write_text(json.dumps({
        "created_at": now(), "python": sys.executable, "source_sha256": hashes,
        "train_command": train_command, "eval_command": eval_command,
        "selection": "best development weighted-F1; held-out test evaluated after training",
    }, indent=2), encoding="utf-8")

    def run_stage(stage, command, log_name):
        with (run_dir / log_name).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_state(status=stage, child_pid=child.pid, active_log=log_name)
            code = child.wait()
        write_state(last_exit_code=code)
        if code:
            raise RuntimeError(f"{stage} exited with code {code}; see {log_name}")

    try:
        run_stage("training", train_command, "train.log")
        run_stage("evaluating_test", eval_command, "evaluate.log")
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "best_model" / "metadata.json").read_text(encoding="utf-8"))
        write_state(status="completed", completed_at=now(), child_pid=None,
                    best_epoch=metadata["best_epoch"],
                    best_dev_weighted_f1=metadata["best_dev_weighted_f1"],
                    test_accuracy=metrics["accuracy"],
                    test_weighted_f1=metrics["weighted_f1"], test_macro_f1=metrics["macro_f1"])
    except BaseException as error:
        write_state(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
