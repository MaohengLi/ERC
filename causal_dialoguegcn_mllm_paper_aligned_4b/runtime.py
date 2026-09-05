"""4-bit QLoRA runtime and two-pass causal evaluation for the aligned model."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from .causal_router import DualSystemDecision, classify_multimodal_dialogue, decisions_to_gate_features
    from .data import LABELS, SPECIAL_TOKENS, build_tokenizer, collate_causal
    from .model import CausalDialogueGCNLLM
except ImportError:
    from causal_router import DualSystemDecision, classify_multimodal_dialogue, decisions_to_gate_features
    from data import LABELS, SPECIAL_TOKENS, build_tokenizer, collate_causal
    from model import CausalDialogueGCNLLM


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_4bit_base(model_name: str, tokenizer_size: int, training: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-4B QLoRA requires a CUDA GPU")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant, torch_dtype=dtype,
        trust_remote_code=True, device_map={"": 0})
    base.resize_token_embeddings(tokenizer_size)
    base.config.use_cache = False
    if not training:
        return base
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"]))


def _kwargs(args, sample: dict) -> dict:
    return {
        "text_dim": sample["text_feats"][0].numel(),
        "audio_dim": sample["audio_feats"][0].numel(),
        "visual_dim": sample["visual_feats"][0].numel(),
        "num_classes": len(LABELS), "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads, "graph_layers": args.graph_layers,
        "graph_window": args.graph_window, "dropout": args.dropout,
        "lambda_hgr": args.lambda_hgr, "lambda_aux": args.lambda_aux,
        "llm_micro_batch": args.llm_micro_batch,
        "causal_graph": not getattr(args, "bidirectional_graph", False),
    }


def build_train_model(args, tokenizer, special_ids, sample: dict):
    llm = load_4bit_base(args.model_name, len(tokenizer), training=True)
    return CausalDialogueGCNLLM(llm, special_ids, **_kwargs(args, sample)).cuda()


def build_eval_model(args, tokenizer, special_ids, sample: dict, checkpoint_dir: Path):
    from peft import PeftModel
    base = load_4bit_base(args.model_name, len(tokenizer), training=False)
    llm = PeftModel.from_pretrained(base, checkpoint_dir / "lora")
    model = CausalDialogueGCNLLM(llm, special_ids, **_kwargs(args, sample)).cuda()
    model.load_non_llm(checkpoint_dir / "non_llm.pt")
    return model


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def dialogue_batches(dialogues: Sequence[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(dialogues), size):
        yield list(dialogues[start:start + size])


def split_route_logits(dialogues: Sequence[dict], logits: torch.Tensor) -> dict[str, torch.Tensor]:
    result, offset = {}, 0
    for dialogue in dialogues:
        n = dialogue["utt_count"]
        result[dialogue["vid"]] = logits[offset:offset + n].detach().float().cpu()
        offset += n
    if offset != logits.size(0):
        raise ValueError("Route-logit count does not match dialogue lengths")
    return result


def decisions_from_cache(cache: Mapping[str, torch.Tensor], base_window: int,
                         max_window: int) -> dict[str, list[DualSystemDecision]]:
    return {vid: classify_multimodal_dialogue(logits, base_window, max_window)
            for vid, logits in cache.items()}


def route_features_for_dialogues(dialogues: Sequence[dict], decisions: Mapping | None,
                                 device: torch.device) -> torch.Tensor:
    parts = []
    for dialogue in dialogues:
        values = None if decisions is None else decisions.get(dialogue["vid"])
        parts.append(decisions_to_gate_features(values, dialogue["utt_count"], device))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_two_pass(model: CausalDialogueGCNLLM, dialogues: Sequence[dict], tokenizer,
                      special_ids: tuple[int, int, int], args) -> dict:
    """First pass predicts route logits; second pass uses causal Prompt and feature gates."""
    model.eval(); y_true, y_pred, alphas = [], [], []
    counts = {"standard": 0, "system2_c1": 0, "system1_c2": 0}
    for step, group in enumerate(dialogue_batches(dialogues, args.dialogue_batch_size), 1):
        first_cpu = collate_causal(group, tokenizer, special_ids, None,
                                    args.max_length, args.base_history_window)
        first = move_tensors(first_cpu, torch.device("cuda"))
        first["route_features"] = route_features_for_dialogues(group, None, first["labels"].device)
        initial = model(first)
        decisions = decisions_from_cache(
            split_route_logits(group, initial["route_logits"]),
            args.base_history_window, args.max_history_window)
        del initial, first

        second_cpu = collate_causal(group, tokenizer, special_ids, decisions,
                                    args.max_length, args.base_history_window)
        second = move_tensors(second_cpu, torch.device("cuda"))
        second["route_features"] = route_features_for_dialogues(
            group, decisions, second["labels"].device)
        output = model(second)
        y_true.extend(second_cpu["labels"].tolist())
        y_pred.extend(output["logits"].argmax(-1).cpu().tolist())
        for values in decisions.values():
            for decision in values:
                counts[decision.route_name] += 1
                alphas.append(decision.alpha_system2)
        print(f"evaluation batch={step}/{math.ceil(len(dialogues)/args.dialogue_batch_size)} "
              f"utterances={len(y_true)}", flush=True)
        del output, second
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_alpha_system2": float(np.mean(alphas)) if alphas else 0.5,
        "route_counts": counts, "num_utterances": len(y_true),
        "per_class": classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                             output_dict=True, zero_division=0),
    }


def print_parameter_report(model: CausalDialogueGCNLLM) -> None:
    mode = "bidirectional" if any(not layer.causal_only for layer in model.encoder.graph_layers) else "past-only causal"
    print("\nNetwork: Causal DialogueGCN-LLM (paper-interface aligned)")
    print(f"  T/A/V projections -> cross-modal attention -> shared DialogueGCN ({mode} graph)")
    print("  graph-to-modality adapters + local T/A/V -> feature gates -> A/V/T projections")
    print("  previous-pass multimodal logits -> Peak-End -> C1/C2 Prompt + feature gate")
    print("  LLM input remains: Causal Prompt + Audio/Visual/Text tokens")
    for name, count in model.parameter_report().items():
        print(f"  {name:20s} {count:>15,}")


def save_checkpoint(model: CausalDialogueGCNLLM, tokenizer, output_dir: Path, metadata: dict) -> None:
    best = output_dir / "best_model"; best.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(best / "lora")
    tokenizer.save_pretrained(best / "tokenizer")
    model.save_non_llm(best / "non_llm.pt")
    (best / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def optimizer_step_count(num_dialogues: int, batch_size: int, grad_accum: int, epochs: int) -> int:
    return math.ceil(math.ceil(num_dialogues / batch_size) / grad_accum) * epochs


__all__ = ["SPECIAL_TOKENS", "build_tokenizer", "build_train_model", "build_eval_model",
           "move_tensors", "dialogue_batches", "split_route_logits", "decisions_from_cache",
           "route_features_for_dialogues", "evaluate_two_pass", "print_parameter_report",
           "save_checkpoint", "optimizer_step_count", "seed_everything"]
