"""IEMOCAP data and five-token prompts for the DialogueGCN dual-system model."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from causal_erc_repro_4b.data import LABELS, DialogueDataset, load_iemocap
try:
    from .causal_router import DualSystemDecision
except ImportError:  # direct script execution
    from causal_router import DualSystemDecision


SPECIAL_TOKENS = (
    "<|audio_feat|>", "<|visual_feat|>", "<|text_feat|>",
    "<|graph_feat|>", "<|causal_route|>",
)


def _clean(text: str) -> str:
    return " ".join(str(text).replace("'", "").replace('"', "").split())


def build_dual_prompt(
    dialogue: dict,
    target_idx: int,
    decision: Optional[DualSystemDecision] = None,
    base_history_window: int = 4,
) -> str:
    speakers, texts = dialogue["speakers"], dialogue["texts"]
    if decision is None:
        start = max(0, target_idx - base_history_window)
        guidance = ""
        peak_line = ""
        route_line = "Reasoning mode: standard multimodal analysis."
    else:
        start = min(target_idx, max(0, decision.history_start))
        if decision.causal_type == 0:  # C1: language/history -> current emotion
            guidance = (
                "Historical dialogue context causes the current emotional state. "
                "Use slow, deliberate System 2 reasoning over speaker-aware history."
            )
            route_line = "Reasoning mode: System 2 / C1 (history-driven)."
            peak_line = ""
        elif decision.causal_type == 1:  # C2: emotion state -> current wording
            guidance = (
                "An existing emotional state causes the current wording. Use fast "
                "System 1 reasoning and focus on multimodal peak and end evidence."
            )
            route_line = "Reasoning mode: System 1 / C2 (emotion-driven)."
            peak_line = ""
            if decision.peak_idx is not None and 0 <= decision.peak_idx < target_idx:
                p = decision.peak_idx
                peak_line = f"Peak evidence: {speakers[p]}: {_clean(texts[p])}"
        else:
            guidance = ""
            peak_line = ""
            route_line = "Reasoning mode: standard multimodal analysis."

    history = [f"{speakers[i]}: {_clean(texts[i])}" for i in range(start, target_idx)]
    target = (
        f"{speakers[target_idx]}: <|audio_feat|> <|visual_feat|> <|text_feat|> "
        f"<|graph_feat|> <|causal_route|> {_clean(texts[target_idx])}"
    )
    label_text = ", ".join(LABELS)
    sections = [
        "Now you are an expert in multimodal conversational emotion analysis.",
        route_line,
    ]
    if guidance:
        sections.append(guidance)
    sections.append("Conversation:\n" + "\n".join(history + [target]))
    if peak_line:
        sections.append(peak_line)
    sections.append(
        f"Please select the emotional label of <{speakers[target_idx]}: "
        f"{_clean(texts[target_idx])}> from [{label_text}]."
    )
    return "\n".join(sections)


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS)
    return tokenizer, ids


def _speaker_ids(dialogue: dict) -> list[int]:
    mapping: dict[str, int] = {}
    result = []
    for speaker in dialogue["speakers"]:
        mapping.setdefault(speaker, len(mapping))
        if mapping[speaker] >= 2:
            raise ValueError("IEMOCAP DialogueGCN expects at most two speakers")
        result.append(mapping[speaker])
    return result


def collate_dual(
    dialogues: Sequence[dict], tokenizer, special_token_ids: tuple[int, ...],
    decisions_by_vid: Optional[Mapping[str, Sequence[DualSystemDecision]]] = None,
    max_length: int = 256, base_history_window: int = 4,
) -> dict:
    if not dialogues:
        raise ValueError("Cannot collate an empty dialogue batch")
    if len(special_token_ids) != 5:
        raise ValueError("DialogueGCN prompts require five special token IDs")
    batch_size = len(dialogues)
    lengths = torch.tensor([d["utt_count"] for d in dialogues], dtype=torch.long)
    max_len = int(lengths.max().item())
    dims = [dialogues[0][key][0].numel() for key in
            ("text_feats", "audio_feats", "visual_feats")]
    text = torch.zeros(batch_size, max_len, dims[0])
    audio = torch.zeros(batch_size, max_len, dims[1])
    visual = torch.zeros(batch_size, max_len, dims[2])
    speakers = torch.zeros(batch_size, max_len, dtype=torch.long)
    encoded, labels, vids, utt_indices = [], [], [], []

    for b, dialogue in enumerate(dialogues):
        n = dialogue["utt_count"]
        text[b, :n] = torch.stack(dialogue["text_feats"])
        audio[b, :n] = torch.stack(dialogue["audio_feats"])
        visual[b, :n] = torch.stack(dialogue["visual_feats"])
        speakers[b, :n] = torch.tensor(_speaker_ids(dialogue), dtype=torch.long)
        decisions = None if decisions_by_vid is None else decisions_by_vid.get(dialogue["vid"])
        for t in range(n):
            decision = None if decisions is None else decisions[t]
            prompt = build_dual_prompt(dialogue, t, decision, base_history_window)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            ids = tokenizer(prompt, truncation=True, max_length=max_length,
                            padding=False, return_tensors="pt")["input_ids"][0]
            for token_id in special_token_ids:
                if int((ids == token_id).sum().item()) != 1:
                    raise ValueError(
                        f"A multimodal token was truncated in {dialogue['vid']}:{t}; "
                        "increase --max_length or reduce history windows."
                    )
            encoded.append(ids)
            labels.append(dialogue["labels"][t])
            vids.append(dialogue["vid"])
            utt_indices.append(t)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    token_max = max(ids.numel() for ids in encoded)
    input_ids = torch.full((len(encoded), token_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    positions = [[] for _ in special_token_ids]
    for row, ids in enumerate(encoded):
        input_ids[row, :ids.numel()] = ids
        attention_mask[row, :ids.numel()] = 1
        for p, token_id in zip(positions, special_token_ids):
            p.append(int((ids == token_id).nonzero(as_tuple=True)[0][0].item()))

    return {
        "text": text, "audio": audio, "visual": visual,
        "speaker_ids": speakers, "lengths": lengths,
        "input_ids": input_ids, "attention_mask": attention_mask,
        "audio_positions": torch.tensor(positions[0], dtype=torch.long),
        "visual_positions": torch.tensor(positions[1], dtype=torch.long),
        "text_positions": torch.tensor(positions[2], dtype=torch.long),
        "graph_positions": torch.tensor(positions[3], dtype=torch.long),
        "route_positions": torch.tensor(positions[4], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "vids": vids, "utt_indices": utt_indices,
    }
