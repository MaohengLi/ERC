"""Train Causal DialogueGCN-LLM on IEMOCAP with a 4-bit Qwen3-4B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

try:
    from .data_dual import build_tokenizer, collate_dual, load_iemocap
    from .runtime import (
        build_train_model, decisions_from_cache, dialogue_batches, evaluate_two_pass,
        move_tensors, optimizer_step_count, print_parameter_report,
        route_features_for_dialogues, save_checkpoint, seed_everything, split_aux_logits,
    )
except ImportError:  # python train.py
    from data_dual import build_tokenizer, collate_dual, load_iemocap
    from runtime import (
        build_train_model, decisions_from_cache, dialogue_batches, evaluate_two_pass,
        move_tensors, optimizer_step_count, print_parameter_report,
        route_features_for_dialogues, save_checkpoint, seed_everything, split_aux_logits,
    )


DEFAULT_DATA = (
    r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN"
    r"\MultiModal_DialogGCN\save\data.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal DialogueGCN-LLM with multimodal dual-system routing")
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", default="outputs/causal_dialoguegcn_qwen3_4b")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--dialogue_batch_size", type=int, default=1)
    parser.add_argument("--llm_micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--base_history_window", type=int, default=4)
    parser.add_argument("--max_history_window", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--graph_window", type=int, default=10)
    parser.add_argument("--bidirectional_graph", action="store_true",
                        help="ablation: permit future utterance graph edges")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_hgr", type=float, default=0.1)
    parser.add_argument("--lambda_aux", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_train_dialogues", type=int, default=0,
                        help="debug only; 0 means the complete training split")
    parser.add_argument("--max_dev_dialogues", type=int, default=0,
                        help="debug only; 0 means the complete development split")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if args.dialogue_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch sizes and gradient accumulation must be positive")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_iemocap(args.data_path)
    train_dialogues = list(data["train"])
    dev_dialogues = list(data["dev"])
    if args.max_train_dialogues:
        train_dialogues = train_dialogues[:args.max_train_dialogues]
    if args.max_dev_dialogues:
        dev_dialogues = dev_dialogues[:args.max_dev_dialogues]
    if not train_dialogues or not dev_dialogues:
        raise ValueError("Training and development splits must not be empty")
    print(f"Data dialogues: train={len(train_dialogues)}, dev={len(dev_dialogues)}, "
          f"test={len(data['test'])}", flush=True)

    tokenizer, special_ids = build_tokenizer(args.model_name)
    model = build_train_model(args, tokenizer, special_ids, train_dialogues[0])
    # Required project convention: print architecture/parameters before metrics.
    print_parameter_report(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = optimizer_step_count(
        len(train_dialogues), args.dialogue_batch_size, args.grad_accum, args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    history, previous_aux = [], {}
    best_f1, stale = -1.0, 0
    rng = np.random.default_rng(args.seed)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        # Previous-epoch model predictions only; no gold labels or future epoch
        # output are exposed to the causal router.
        decisions = None if not previous_aux else decisions_from_cache(
            previous_aux, args.base_history_window, args.max_history_window)
        shuffled = [train_dialogues[i] for i in rng.permutation(len(train_dialogues))]
        batches = list(dialogue_batches(shuffled, args.dialogue_batch_size))
        current_aux: dict[str, torch.Tensor] = {}
        total = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "hgr": 0.0}
        train_true, train_pred = [], []
        model.train()

        for step, group in enumerate(batches, start=1):
            cpu = collate_dual(
                group, tokenizer, special_ids, decisions,
                max_length=args.max_length, base_history_window=args.base_history_window,
            )
            batch = move_tensors(cpu, torch.device("cuda"))
            batch["route_features"] = route_features_for_dialogues(
                group, decisions, batch["labels"].device)
            output = model(batch, labels=batch["labels"])
            (output["loss"] / args.grad_accum).backward()
            total["loss"] += float(output["loss"])
            total["ce"] += float(output["loss_ce"])
            total["aux"] += float(output["loss_aux"])
            total["hgr"] += float(output["loss_hgr"])
            current_aux.update(split_aux_logits(group, output["aux_logits"]))
            train_true.extend(cpu["labels"].tolist())
            train_pred.extend(output["logits"].detach().argmax(-1).cpu().tolist())

            if step % args.grad_accum == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0 or step == len(batches):
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"epoch={epoch}/{args.epochs} step={step}/{len(batches)} "
                      f"loss={total['loss']/step:.4f} gpu_peak={mem:.2f}GB", flush=True)

        previous_aux = current_aux
        dev = evaluate_two_pass(model, dev_dialogues, tokenizer, special_ids, args)
        record = {
            "epoch": epoch, "prompt_mode": "standard" if epoch == 1 else "dual_system",
            **{f"train_{k}": v / len(batches) for k, v in total.items()},
            "train_accuracy": float(accuracy_score(train_true, train_pred)),
            "train_weighted_f1": float(f1_score(
                train_true, train_pred, average="weighted", zero_division=0)),
            **{f"dev_{k}": v for k, v in dev.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if dev["weighted_f1"] > best_f1 + 1e-4:
            best_f1, stale = dev["weighted_f1"], 0
            save_checkpoint(model, tokenizer, output_dir, {
                "architecture": "Causal DialogueGCN-LLM with Multimodal Dual-System Routing",
                "model_name": args.model_name, "best_epoch": epoch,
                "best_dev_weighted_f1": best_f1, "args": vars(args),
            })
        else:
            stale += 1
        (output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    print_parameter_report(model)
    print(f"Training finished: best dev weighted-F1={best_f1:.4f}", flush=True)
    print(f"Best checkpoint: {output_dir / 'best_model'}", flush=True)


if __name__ == "__main__":
    train(parse_args())
