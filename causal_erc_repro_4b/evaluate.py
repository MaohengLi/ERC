"""Load the best 4B reproduction checkpoint and run two-pass causal evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .data import load_iemocap
    from .runtime import build_eval_model, build_tokenizer, evaluate_two_pass, print_parameter_report
    from .train import DEFAULT_DATA
except ImportError:
    from data import load_iemocap
    from runtime import build_eval_model, build_tokenizer, evaluate_two_pass, print_parameter_report
    from train import DEFAULT_DATA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--checkpoint_dir", default="outputs/qwen3_4b/best_model")
    parser.add_argument("--dialogue_batch_size", type=int, default=1)
    parser.add_argument("--llm_micro_batch", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--base_history_window", type=int, default=4)
    parser.add_argument("--max_history_window", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_hgr", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/qwen3_4b/test_results.json")
    return parser.parse_args()


def main(args):
    data = load_iemocap(args.data_path)
    tokenizer, special_ids = build_tokenizer(args.model_name)
    model = build_eval_model(
        args, tokenizer, special_ids, data["train"][0], Path(args.checkpoint_dir).resolve())
    print_parameter_report(model)
    metrics = evaluate_two_pass(model, data["test"], tokenizer, special_ids, args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Test results")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
