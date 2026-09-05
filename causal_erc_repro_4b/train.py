"""Train the paper-aligned Causal-ERC reproduction with Qwen3-4B QLoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from .data import collate_dialogues, load_iemocap
    from .runtime import (
        build_tokenizer, build_train_model, decisions_from_cache, dialogue_batches,
        evaluate_two_pass, move_tensors, optimizer_step_count, print_parameter_report,
        save_checkpoint, seed_everything, split_logits,
    )
except ImportError:
    from data import collate_dialogues, load_iemocap
    from runtime import (
        build_tokenizer, build_train_model, decisions_from_cache, dialogue_batches,
        evaluate_two_pass, move_tensors, optimizer_step_count, print_parameter_report,
        save_checkpoint, seed_everything, split_logits,
    )


DEFAULT_DATA = (
    r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN"
    r"\MultiModal_DialogGCN\save\data.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal-ERC reproduction on a 4B Qwen model")
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", default="outputs/qwen3_4b")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--dialogue_batch_size", type=int, default=1)
    parser.add_argument("--llm_micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--base_history_window", type=int, default=4)
    parser.add_argument("--max_history_window", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_hgr", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_dialogues", type=int, default=0,
                        help="debug-only limit; 0 uses the full training split")
    parser.add_argument("--max_dev_dialogues", type=int, default=0,
                        help="debug-only limit; 0 uses the full development split")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_iemocap(args.data_path)
    train_dialogues = list(data["train"])
    dev_dialogues = list(data["dev"])
    if args.max_train_dialogues > 0:
        train_dialogues = train_dialogues[:args.max_train_dialogues]
    if args.max_dev_dialogues > 0:
        dev_dialogues = dev_dialogues[:args.max_dev_dialogues]
    print(f"Data: train={len(train_dialogues)}, dev={len(dev_dialogues)}, test={len(data['test'])} dialogues")

    tokenizer, special_ids = build_tokenizer(args.model_name)
    model = build_train_model(args, tokenizer, special_ids, train_dialogues[0])
    print_parameter_report(model)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = optimizer_step_count(
        len(train_dialogues), args.dialogue_batch_size, args.grad_accum, args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1, bad_epochs = -1.0, 0
    previous_logits: dict[str, torch.Tensor] = {}
    history = []
    rng = np.random.default_rng(args.seed)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        # Crucially keyed by dialogue ID: shuffle cannot corrupt causal assignments.
        decisions = None if not previous_logits else decisions_from_cache(
            previous_logits, args.base_history_window, args.max_history_window)
        order = rng.permutation(len(train_dialogues))
        shuffled = [train_dialogues[i] for i in order]
        current_logits: dict[str, torch.Tensor] = {}
        model.train()
        total_loss = total_ce = total_hgr = 0.0
        batches = list(dialogue_batches(shuffled, args.dialogue_batch_size))

        for step, dialogues in enumerate(batches, start=1):
            batch_cpu = collate_dialogues(
                dialogues, tokenizer, special_ids, decisions,
                max_length=args.max_length,
                base_history_window=args.base_history_window,
            )
            batch = move_tensors(batch_cpu, torch.device("cuda"))
            output = model(batch, labels=batch["labels"])
            (output["loss"] / args.grad_accum).backward()
            total_loss += float(output["loss"].item())
            total_ce += float(output["loss_ce"].item())
            total_hgr += float(output["loss_hgr"].item())
            current_logits.update(split_logits(dialogues, output["logits"]))

            if step % args.grad_accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                print(f"Epoch {epoch} step {step}/{len(batches)} loss={total_loss/step:.4f}", flush=True)

        previous_logits = current_logits
        dev_metrics = evaluate_two_pass(model, dev_dialogues, tokenizer, special_ids, args)
        epoch_record = {
            "epoch": epoch,
            "prompt": "standard" if epoch == 1 else "causal",
            "train_loss": total_loss / len(batches),
            "train_ce": total_ce / len(batches),
            "train_hgr": total_hgr / len(batches),
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        if dev_metrics["weighted_f1"] > best_f1 + 1e-4:
            best_f1 = dev_metrics["weighted_f1"]
            bad_epochs = 0
            save_checkpoint(model, tokenizer, output_dir, {
                "model_name": args.model_name,
                "best_epoch": epoch,
                "best_dev_weighted_f1": best_f1,
                "args": vars(args),
            })
        else:
            bad_epochs += 1
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        if bad_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}")
            break

    # Repository convention: architecture first, then final training result.
    print_parameter_report(model)
    print(f"Training result: best dev weighted-F1={best_f1:.4f}")
    print("Run evaluate.py to load the best checkpoint and evaluate the held-out test split.")


if __name__ == "__main__":
    train(parse_args())
