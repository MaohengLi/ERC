"""IEMOCAP dialogue loader and leak-free batch/prompt collation."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .causal_prompting import CausalDecision, build_erc_prompt
except ImportError:  # direct script execution from this directory
    from causal_prompting import CausalDecision, build_erc_prompt


LABELS = ["happiness", "sadness", "neutral", "anger", "excitement", "frustration"]
LABEL_ALIASES = {
    "hap": 0, "happiness": 0, "happy": 0,
    "sad": 1, "sadness": 1,
    "neu": 2, "neutral": 2,
    "ang": 3, "anger": 3, "angry": 3,
    "exc": 4, "excitement": 4, "excited": 4,
    "fru": 5, "frustration": 5, "frustrated": 5,
}


def _label_index(value) -> int:
    if isinstance(value, (int, np.integer)):
        idx = int(value)
    else:
        idx = LABEL_ALIASES.get(str(value).lower(), -1)
    if not 0 <= idx < len(LABELS):
        raise ValueError(f"Unknown IEMOCAP label: {value!r}")
    return idx


def load_iemocap(data_path: str | Path) -> dict[str, list[dict]]:
    """Load the existing MultiModal_DialogGCN ``data.pkl`` without modifying it."""

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))  # makes dgcn.Sample importable for pickle

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IEMOCAP pickle not found: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream)

    result: dict[str, list[dict]] = {}
    for split in ("train", "dev", "test"):
        result[split] = []
        for sample in raw[split]:
            n = len(sample.text)
            if not all(len(getattr(sample, name)) == n for name in
                       ("audio", "visual", "speaker", "label", "sentence")):
                raise ValueError(f"Misaligned modalities in dialogue {sample.vid}")
            result[split].append({
                "vid": str(sample.vid),
                "utt_count": n,
                "text_feats": [torch.as_tensor(x, dtype=torch.float32) for x in sample.text],
                "audio_feats": [torch.as_tensor(x, dtype=torch.float32) for x in sample.audio],
                "visual_feats": [torch.as_tensor(x, dtype=torch.float32) for x in sample.visual],
                "speakers": [str(x) for x in sample.speaker],
                "labels": [_label_index(x) for x in sample.label],
                "texts": [str(x) for x in sample.sentence],
            })
    return result


class DialogueDataset(Dataset):
    def __init__(self, dialogues: Sequence[dict]):
        self.dialogues = list(dialogues)

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, index: int) -> dict:
        return self.dialogues[index]


def _speaker_ids(dialogue: dict, max_speakers: int = 16) -> list[int]:
    mapping: dict[str, int] = {}
    ids = []
    for speaker in dialogue["speakers"]:
        if speaker not in mapping:
            mapping[speaker] = len(mapping)
        if mapping[speaker] >= max_speakers:
            raise ValueError(f"Dialogue {dialogue['vid']} exceeds {max_speakers} speakers")
        ids.append(mapping[speaker])
    return ids


def collate_dialogues(
    dialogues: Sequence[dict],
    tokenizer,
    special_token_ids: tuple[int, int, int],
    decisions_by_vid: Optional[Mapping[str, Sequence[CausalDecision]]] = None,
    max_length: int = 256,
    base_history_window: int = 4,
) -> dict[str, torch.Tensor | list[str]]:
    """Pad dialogue features and build one causal ERC prompt per utterance."""

    if not dialogues:
        raise ValueError("Cannot collate an empty dialogue batch")
    a_id, v_id, t_id = special_token_ids
    batch_size = len(dialogues)
    lengths = torch.tensor([d["utt_count"] for d in dialogues], dtype=torch.long)
    max_utts = int(lengths.max().item())
    text_dim = dialogues[0]["text_feats"][0].numel()
    audio_dim = dialogues[0]["audio_feats"][0].numel()
    video_dim = dialogues[0]["visual_feats"][0].numel()

    text = torch.zeros(batch_size, max_utts, text_dim)
    audio = torch.zeros(batch_size, max_utts, audio_dim)
    visual = torch.zeros(batch_size, max_utts, video_dim)
    speakers = torch.zeros(batch_size, max_utts, dtype=torch.long)

    encoded, labels, vids, utt_indices = [], [], [], []
    for b, dialogue in enumerate(dialogues):
        n = dialogue["utt_count"]
        text[b, :n] = torch.stack(dialogue["text_feats"])
        audio[b, :n] = torch.stack(dialogue["audio_feats"])
        visual[b, :n] = torch.stack(dialogue["visual_feats"])
        speakers[b, :n] = torch.tensor(_speaker_ids(dialogue), dtype=torch.long)
        dialogue_decisions = None if decisions_by_vid is None else decisions_by_vid.get(dialogue["vid"])

        for t in range(n):
            decision = None if dialogue_decisions is None else dialogue_decisions[t]
            prompt = build_erc_prompt(
                dialogue["speakers"], dialogue["texts"], t, LABELS,
                decision=decision, base_history_window=base_history_window,
            )
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True,
                )
            item = tokenizer(prompt, truncation=True, max_length=max_length,
                             padding=False, return_tensors="pt")["input_ids"][0]
            for token_id, name in zip((a_id, v_id, t_id), ("audio", "visual", "text")):
                count = int((item == token_id).sum().item())
                if count != 1:
                    raise ValueError(
                        f"Expected one {name} token for {dialogue['vid']}:{t}, found {count}. "
                        "Increase --max_length or reduce history windows."
                    )
            encoded.append(item)
            labels.append(dialogue["labels"][t])
            vids.append(dialogue["vid"])
            utt_indices.append(t)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    token_max = max(x.numel() for x in encoded)
    input_ids = torch.full((len(encoded), token_max), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), token_max), dtype=torch.long)
    positions = [[], [], []]
    for row, ids in enumerate(encoded):
        input_ids[row, :ids.numel()] = ids
        attention_mask[row, :ids.numel()] = 1
        for target, token_id in zip(positions, (a_id, v_id, t_id)):
            target.append(int((ids == token_id).nonzero(as_tuple=True)[0][0].item()))

    return {
        "text": text, "audio": audio, "visual": visual,
        "speaker_ids": speakers, "lengths": lengths,
        "input_ids": input_ids, "attention_mask": attention_mask,
        "audio_positions": torch.tensor(positions[0], dtype=torch.long),
        "visual_positions": torch.tensor(positions[1], dtype=torch.long),
        "text_positions": torch.tensor(positions[2], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "vids": vids, "utt_indices": utt_indices,
    }
