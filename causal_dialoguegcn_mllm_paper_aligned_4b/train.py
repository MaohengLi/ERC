"""Train the paper-interface-aligned DialogueGCN Causal-ERC extension."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

try:
    from .data import build_tokenizer, collate_causal, load_iemocap
    from .runtime import (build_train_model, decisions_from_cache, dialogue_batches,
                          evaluate_two_pass, move_tensors, optimizer_step_count,
                          print_parameter_report, route_features_for_dialogues,
                          save_checkpoint, seed_everything, split_route_logits)
except ImportError:
    from data import build_tokenizer, collate_causal, load_iemocap
    from runtime import (build_train_model, decisions_from_cache, dialogue_batches,
                         evaluate_two_pass, move_tensors, optimizer_step_count,
                         print_parameter_report, route_features_for_dialogues,
                         save_checkpoint, seed_everything, split_route_logits)


DEFAULT_DATA = (r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN"
                r"\MultiModal_DialogGCN\save\data.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal DialogueGCN-LLM (Causal-ERC interface aligned)")
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", default="outputs/causal_dialoguegcn_aligned_qwen3_4b")
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
                        help="ablation only: permit future graph edges")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_hgr", type=float, default=1.0)
    parser.add_argument("--lambda_aux", type=float, default=0.1,
                        help="internal T/A/V router probe supervision")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_train_dialogues", type=int, default=0)
    parser.add_argument("--max_dev_dialogues", type=int, default=0)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if args.dialogue_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("dialogue_batch_size and grad_accum must be positive")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    data = load_iemocap(args.data_path)
    train_dialogues, dev_dialogues = list(data["train"]), list(data["dev"])
    if args.max_train_dialogues:
        train_dialogues = train_dialogues[:args.max_train_dialogues]
    if args.max_dev_dialogues:
        dev_dialogues = dev_dialogues[:args.max_dev_dialogues]
    if not train_dialogues or not dev_dialogues:
        raise ValueError("train/dev split cannot be empty")
    print(f"Data: train={len(train_dialogues)}, dev={len(dev_dialogues)}, "
          f"test={len(data['test'])}", flush=True)
    (output_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "splits": {
            name: {"dialogues": len(values), "utterances": sum(d["utt_count"] for d in values)}
            for name, values in (("train", train_dialogues), ("dev", dev_dialogues), ("test", data["test"]))
        }}, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer, special_ids = build_tokenizer(args.model_name)
    model = build_train_model(args, tokenizer, special_ids, train_dialogues[0])
    print_parameter_report(model)
    (output_dir / "parameter_counts.json").write_text(
        json.dumps(model.parameter_report(), indent=2), encoding="utf-8")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = optimizer_step_count(len(train_dialogues), args.dialogue_batch_size,
                                       args.grad_accum, args.epochs)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * args.warmup_ratio)), total_steps)

    rng = np.random.default_rng(args.seed)
    previous_cache: dict[str, torch.Tensor] = {}
    history, best_f1, stale = [], -1.0, 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        processed_utts = 0
        decisions = None if not previous_cache else decisions_from_cache(
            previous_cache, args.base_history_window, args.max_history_window)
        shuffled = [train_dialogues[i] for i in rng.permutation(len(train_dialogues))]
        groups = list(dialogue_batches(shuffled, args.dialogue_batch_size))
        current_cache: dict[str, torch.Tensor] = {}
        totals = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "hgr": 0.0}
        y_true, y_pred = [], []
        model.train()

        for step, group in enumerate(groups, 1):
            cpu = collate_causal(group, tokenizer, special_ids, decisions,
                                 args.max_length, args.base_history_window)
            batch = move_tensors(cpu, torch.device("cuda"))
            batch["route_features"] = route_features_for_dialogues(
                group, decisions, batch["labels"].device)
            output = model(batch, labels=batch["labels"])
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
            accumulation_size = min(args.grad_accum, len(groups) - ((step - 1) // args.grad_accum) * args.grad_accum)
            (output["loss"] / accumulation_size).backward()
            for key, out_key in (("loss", "loss"), ("ce", "loss_ce"),
                                 ("aux", "loss_aux"), ("hgr", "loss_hgr")):
                totals[key] += float(output[out_key]) if out_key in output else 0.0
            current_cache.update(split_route_logits(group, output["route_logits"]))
            y_true.extend(cpu["labels"].tolist())
            y_pred.extend(output["logits"].detach().argmax(-1).cpu().tolist())
            processed_utts += len(cpu["labels"])
            if step % args.grad_accum == 0 or step == len(groups):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0 or step == len(groups):
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"epoch={epoch}/{args.epochs} step={step}/{len(groups)} "
                      f"loss={totals['loss']/step:.4f} gpu_peak={mem:.2f}GB "
                      f"utterances={processed_utts} elapsed_s={time.monotonic()-epoch_started:.1f}", flush=True)
                (output_dir / "progress.json").write_text(json.dumps({
                    "stage": "training", "epoch": epoch, "batch": step, "batches": len(groups),
                    "utterances": processed_utts, "elapsed_s": time.monotonic() - epoch_started,
                    "loss": totals["loss"] / step, "gpu_peak_gb": mem,
                }, indent=2), encoding="utf-8")
            del output, batch

        previous_cache = current_cache
        print(f"epoch={epoch} validation starting (two-pass)", flush=True)
        dev = evaluate_two_pass(model, dev_dialogues, tokenizer, special_ids, args)
        record = {
            "epoch": epoch, "prompt_mode": "standard" if epoch == 1 else "causal_dual_system",
            **{f"train_{k}": value / len(groups) for k, value in totals.items()},
            "train_accuracy": float(accuracy_score(y_true, y_pred)),
            "train_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            **{f"dev_{k}": value for k, value in dev.items()},
        }
        history.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
        if dev["weighted_f1"] > best_f1 + 1e-4:
            best_f1, stale = dev["weighted_f1"], 0
            save_checkpoint(model, tokenizer, output_dir, {
                "architecture": "Causal-ERC -> DialogueGCN + multimodal Peak-End dual-system routing",
                "llm_interface": "Causal Prompt + Audio/Visual/Text feature tokens",
                "best_epoch": epoch, "best_dev_weighted_f1": best_f1,
                "model_name": args.model_name, "args": vars(args),
            })
        else:
            stale += 1
        (output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True); break

    print_parameter_report(model)
    print(f"Training finished: best dev weighted-F1={best_f1:.4f}", flush=True)
    print(f"Best checkpoint: {output_dir / 'best_model'}", flush=True)


if __name__ == "__main__":
    train(parse_args())
