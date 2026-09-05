"""Runtime, QLoRA loading, checkpointing and two-pass evaluation utilities."""

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
    from .causal_router import DualSystemDecision, classify_multimodal_dialogue, decisions_to_features
    from .data_dual import LABELS, SPECIAL_TOKENS, build_tokenizer, collate_dual
    from .model import CausalDialogueGCNLLM
except ImportError:  # direct script execution
    from causal_router import DualSystemDecision, classify_multimodal_dialogue, decisions_to_features
    from data_dual import LABELS, SPECIAL_TOKENS, build_tokenizer, collate_dual
    from model import CausalDialogueGCNLLM


TOKEN_NAMES = ("audio", "visual", "text", "graph", "route")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_4bit_base(model_name: str, tokenizer_size: int, training: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-4B QLoRA requires a CUDA GPU")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant, torch_dtype=dtype,
        trust_remote_code=True, device_map={"": 0},
    )
    base.resize_token_embeddings(tokenizer_size)
    base.config.use_cache = False
    if not training:
        return base

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(base, config)


def _token_id_map(special_ids: tuple[int, ...]) -> dict[str, int]:
    if len(special_ids) != len(TOKEN_NAMES):
        raise ValueError("Five special token IDs are required")
    return dict(zip(TOKEN_NAMES, special_ids))


def _model_kwargs(args, sample: dict) -> dict:
    return {
        "text_dim": sample["text_feats"][0].numel(),
        "audio_dim": sample["audio_feats"][0].numel(),
        "visual_dim": sample["visual_feats"][0].numel(),
        "num_classes": len(LABELS), "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads, "graph_layers": args.graph_layers,
        "graph_window": args.graph_window, "dropout": args.dropout,
        "causal_graph": not getattr(args, "bidirectional_graph", False),
        "lambda_hgr": args.lambda_hgr, "lambda_aux": args.lambda_aux,
        "llm_micro_batch": args.llm_micro_batch,
    }


def build_train_model(args, tokenizer, special_ids: tuple[int, ...], sample: dict):
    llm = load_4bit_base(args.model_name, len(tokenizer), training=True)
    return CausalDialogueGCNLLM(
        llm, _token_id_map(special_ids), **_model_kwargs(args, sample)).cuda()


def build_eval_model(args, tokenizer, special_ids: tuple[int, ...], sample: dict,
                     checkpoint_dir: Path):
    from peft import PeftModel
    base = load_4bit_base(args.model_name, len(tokenizer), training=False)
    llm = PeftModel.from_pretrained(base, checkpoint_dir / "lora")
    model = CausalDialogueGCNLLM(
        llm, _token_id_map(special_ids), **_model_kwargs(args, sample)).cuda()
    model.load_non_llm(checkpoint_dir / "non_llm.pt")
    return model


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def dialogue_batches(dialogues: Sequence[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(dialogues), size):
        yield list(dialogues[start:start + size])


def split_aux_logits(dialogues: Sequence[dict], logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split [all utterances, 4, classes] cache by immutable dialogue ID."""
    result, offset = {}, 0
    for dialogue in dialogues:
        n = dialogue["utt_count"]
        result[dialogue["vid"]] = logits[offset:offset + n].detach().float().cpu()
        offset += n
    if offset != logits.size(0):
        raise ValueError("Auxiliary logit count does not match dialogue lengths")
    return result


def decisions_from_cache(cache: Mapping[str, torch.Tensor], base_window: int,
                         max_window: int) -> dict[str, list[DualSystemDecision]]:
    return {vid: classify_multimodal_dialogue(value, base_window, max_window)
            for vid, value in cache.items()}


def route_features_for_dialogues(dialogues: Sequence[dict],
                                 decisions: Mapping[str, Sequence[DualSystemDecision]] | None,
                                 device: torch.device | None = None) -> torch.Tensor:
    parts = []
    for dialogue in dialogues:
        values = None if decisions is None else decisions.get(dialogue["vid"])
        parts.append(decisions_to_features(values, dialogue["utt_count"], device=device))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_two_pass(model: CausalDialogueGCNLLM, dialogues: Sequence[dict], tokenizer,
                      special_ids: tuple[int, ...], args) -> dict:
    """Leak-free inference: standard pass -> cached multimodal route -> causal pass."""
    model.eval()
    y_true, y_pred = [], []
    type_counts = {"standard": 0, "system2_c1": 0, "system1_c2": 0}
    alpha_values = []
    for group in dialogue_batches(dialogues, args.dialogue_batch_size):
        first_cpu = collate_dual(
            group, tokenizer, special_ids, decisions_by_vid=None,
            max_length=args.max_length, base_history_window=args.base_history_window,
        )
        first = move_tensors(first_cpu, torch.device("cuda"))
        first["route_features"] = route_features_for_dialogues(group, None, first["labels"].device)
        initial = model(first)
        cache = split_aux_logits(group, initial["aux_logits"])
        decisions = decisions_from_cache(cache, args.base_history_window, args.max_history_window)

        second_cpu = collate_dual(
            group, tokenizer, special_ids, decisions,
            max_length=args.max_length, base_history_window=args.base_history_window,
        )
        second = move_tensors(second_cpu, torch.device("cuda"))
        second["route_features"] = route_features_for_dialogues(
            group, decisions, second["labels"].device)
        output = model(second)
        y_true.extend(second_cpu["labels"].tolist())
        y_pred.extend(output["logits"].argmax(dim=-1).cpu().tolist())
        for values in decisions.values():
            for decision in values:
                type_counts[decision.route_name] += 1
                alpha_values.append(decision.alpha_system2)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_alpha_system2": float(np.mean(alpha_values)) if alpha_values else 0.5,
        "route_counts": type_counts, "num_utterances": len(y_true),
        "per_class": classification_report(y_true, y_pred, target_names=LABELS,
                                             output_dict=True, zero_division=0),
    }


def print_parameter_report(model: CausalDialogueGCNLLM) -> None:
    print("\nNetwork: Causal DialogueGCN-LLM with Multimodal Dual-System Routing")
    graph_mode = "bidirectional" if any(
        not layer.causal_only for layer in model.encoder.graph_layers) else "past-only causal"
    print(f"  raw T/A/V -> cross-modal attention -> relational DialogueGCN ({graph_mode})")
    print("  previous-pass T/A/V/G logits -> Peak-End C1/C2 -> System-2/System-1 soft route")
    print("  five feature tokens -> Qwen3-4B NF4 + LoRA -> 6-way emotion classifier")
    print("  loss = CE + lambda_aux * auxiliary_CE + lambda_hgr * Soft-HGR")
    for name, count in model.parameter_report().items():
        print(f"  {name:20s} {count:>15,}")


def save_checkpoint(model: CausalDialogueGCNLLM, tokenizer, output_dir: Path,
                    metadata: dict) -> None:
    best = output_dir / "best_model"
    best.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(best / "lora")
    tokenizer.save_pretrained(best / "tokenizer")
    model.save_non_llm(best / "non_llm.pt")
    (best / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def optimizer_step_count(num_dialogues: int, batch_size: int,
                         grad_accum: int, epochs: int) -> int:
    batches = math.ceil(num_dialogues / batch_size)
    return math.ceil(batches / grad_accum) * epochs


__all__ = [
    "SPECIAL_TOKENS", "build_tokenizer", "build_train_model", "build_eval_model",
    "move_tensors", "dialogue_batches", "split_aux_logits", "decisions_from_cache",
    "route_features_for_dialogues", "evaluate_two_pass", "print_parameter_report",
    "save_checkpoint", "optimizer_step_count", "seed_everything",
]
