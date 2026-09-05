"""Evaluate a paper-interface-aligned checkpoint using two-pass routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .data import build_tokenizer, load_iemocap
    from .runtime import build_eval_model, evaluate_two_pass, print_parameter_report, seed_everything
except ImportError:
    from data import build_tokenizer, load_iemocap
    from runtime import build_eval_model, evaluate_two_pass, print_parameter_report, seed_everything


DEFAULT_DATA = (r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN"
                r"\MultiModal_DialogGCN\save\data.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aligned DialogueGCN Causal-ERC")
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--checkpoint_dir", default="outputs/causal_dialoguegcn_aligned_qwen3_4b/best_model")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--dialogue_batch_size", type=int, default=1)
    parser.add_argument("--llm_micro_batch", type=int, default=2)
    parser.add_argument("--max_dialogues", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint_dir).resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    saved = metadata["args"]
    for key in ("model_name", "hidden_dim", "num_heads", "graph_layers", "graph_window",
                "bidirectional_graph", "dropout", "lambda_hgr", "lambda_aux",
                "base_history_window", "max_history_window", "max_length"):
        setattr(args, key, saved[key])
    seed_everything(args.seed)
    data = load_iemocap(args.data_path); dialogues = list(data[args.split])
    if args.max_dialogues:
        dialogues = dialogues[:args.max_dialogues]
    tokenizer, special_ids = build_tokenizer(str(checkpoint / "tokenizer"))
    model = build_eval_model(args, tokenizer, special_ids, dialogues[0], checkpoint)
    print_parameter_report(model)
    result = {"split": args.split, "checkpoint": str(checkpoint), **evaluate_two_pass(
        model, dialogues, tokenizer, special_ids, args)}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    out = Path(args.output_json).resolve() if args.output_json else checkpoint / f"{args.split}_metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate(parse_args())

