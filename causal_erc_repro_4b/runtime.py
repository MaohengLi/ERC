"""Shared 4-bit Qwen loading, checkpointing, and evaluation utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

try:
    from .causal_prompting import CausalDecision, classify_dialogue
    from .data import LABELS, collate_dialogues
    from .model import CausalERC4B
except ImportError:
    from causal_prompting import CausalDecision, classify_dialogue
    from data import LABELS, collate_dialogues
    from model import CausalERC4B


SPECIAL_TOKENS = ("<|audio_feat|>", "<|visual_feat|>", "<|text_feat|>")


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Keep the target utterance and its modality tokens if truncation is necessary.
    tokenizer.truncation_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS)
    return tokenizer, ids


def load_4bit_base(model_name: str, tokenizer_size: int, training: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("The 4-bit Causal-ERC reproduction requires a CUDA GPU")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, torch_dtype=compute_dtype,
        trust_remote_code=True, device_map={"": 0},
    )
    base.resize_token_embeddings(tokenizer_size)
    base.config.use_cache = False
    if training:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        return get_peft_model(base, lora)
    return base


def build_train_model(args, tokenizer, special_ids, sample: dict) -> CausalERC4B:
    llm = load_4bit_base(args.model_name, len(tokenizer), training=True)
    model = CausalERC4B(
        llm, special_ids,
        text_dim=sample["text_feats"][0].numel(),
        audio_dim=sample["audio_feats"][0].numel(),
        video_dim=sample["visual_feats"][0].numel(),
        hidden_dim=args.hidden_dim,
        num_classes=len(LABELS), num_heads=args.num_heads,
        dropout=args.dropout, lambda_hgr=args.lambda_hgr,
        llm_micro_batch=args.llm_micro_batch,
    )
    return model.cuda()


def build_eval_model(args, tokenizer, special_ids, sample: dict, checkpoint_dir: Path) -> CausalERC4B:
    from peft import PeftModel
    base = load_4bit_base(args.model_name, len(tokenizer), training=False)
    llm = PeftModel.from_pretrained(base, checkpoint_dir / "lora")
    model = CausalERC4B(
        llm, special_ids,
        text_dim=sample["text_feats"][0].numel(),
        audio_dim=sample["audio_feats"][0].numel(),
        video_dim=sample["visual_feats"][0].numel(),
        hidden_dim=args.hidden_dim,
        num_classes=len(LABELS), num_heads=args.num_heads,
        dropout=args.dropout, lambda_hgr=args.lambda_hgr,
        llm_micro_batch=args.llm_micro_batch,
    ).cuda()
    model.load_non_llm(checkpoint_dir / "non_llm.pt")
    return model


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def dialogue_batches(dialogues: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(dialogues), batch_size):
        yield list(dialogues[start:start + batch_size])


def split_logits(dialogues: Sequence[dict], logits: torch.Tensor) -> dict[str, torch.Tensor]:
    result, offset = {}, 0
    for dialogue in dialogues:
        n = dialogue["utt_count"]
        result[dialogue["vid"]] = logits[offset:offset + n].detach().cpu()
        offset += n
    if offset != logits.size(0):
        raise ValueError("Logit count does not match dialogue lengths")
    return result


def decisions_from_cache(
    cache: Mapping[str, torch.Tensor],
    base_history_window: int,
    max_history_window: int,
) -> dict[str, list[CausalDecision]]:
    return {
        vid: classify_dialogue(logits, base_history_window, max_history_window)
        for vid, logits in cache.items()
    }


@torch.no_grad()
def evaluate_two_pass(model: CausalERC4B, dialogues: Sequence[dict], tokenizer,
                      special_ids: tuple[int, int, int], args) -> dict:
    """Paper-style inference: standard pass, then Peak-End causal re-prompting."""

    model.eval()
    y_true, y_pred, type_counts = [], [], {"standard": 0, "c1": 0, "c2": 0}
    for batch_dialogues in dialogue_batches(dialogues, args.dialogue_batch_size):
        standard_cpu = collate_dialogues(
            batch_dialogues, tokenizer, special_ids, max_length=args.max_length,
            base_history_window=args.base_history_window,
        )
        standard = move_tensors(standard_cpu, torch.device("cuda"))
        first = model(standard)
        first_cache = split_logits(batch_dialogues, first["logits"])
        decisions = decisions_from_cache(
            first_cache, args.base_history_window, args.max_history_window)
        causal_cpu = collate_dialogues(
            batch_dialogues, tokenizer, special_ids, decisions,
            max_length=args.max_length,
            base_history_window=args.base_history_window,
        )
        causal = move_tensors(causal_cpu, torch.device("cuda"))
        second = model(causal)
        y_true.extend(causal_cpu["labels"].tolist())
        y_pred.extend(second["logits"].argmax(-1).cpu().tolist())
        for values in decisions.values():
            for decision in values:
                key = "standard" if decision.causal_type is None else f"c{decision.causal_type + 1}"
                type_counts[key] += 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "causal_type_counts": type_counts,
        "num_utterances": len(y_true),
    }


def print_parameter_report(model: CausalERC4B) -> None:
    print("\nNetwork architecture and parameter counts")
    for name, count in model.parameter_report().items():
        print(f"  {name:20s} {count:>15,}")


def save_checkpoint(model: CausalERC4B, tokenizer, output_dir: Path,
                    metadata: dict) -> None:
    best = output_dir / "best_model"
    best.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(best / "lora")
    tokenizer.save_pretrained(best / "tokenizer")
    model.save_non_llm(best / "non_llm.pt")
    (best / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def optimizer_step_count(num_dialogues: int, dialogue_batch_size: int,
                         grad_accum: int, epochs: int) -> int:
    batches = math.ceil(num_dialogues / dialogue_batch_size)
    return math.ceil(batches / grad_accum) * epochs
