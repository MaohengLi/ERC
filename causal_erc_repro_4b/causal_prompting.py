"""Peak-End causal categorisation and prompt construction from Causal-ERC Sec. 3.5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


C1_PROMPT = (
    "In this utterance, historical utterances cause the speaker's emotion. "
    "The speaker then produces the current utterance. Therefore, carefully "
    "consider the speaker-aware dialogue history."
)
C2_PROMPT = (
    "In this utterance, the speaker first experiences an emotion, which then "
    "leads to the current utterance. Therefore, focus on the emotional peak "
    "utterance and the most recent utterance to infer the underlying emotion."
)


@dataclass(frozen=True)
class CausalDecision:
    """Decision computed only from the previous epoch's logits of one dialogue."""

    causal_type: Optional[int]  # None=standard, 0=C1/System 2, 1=C2/System 1
    lambda_c1: float = 0.0
    lambda_c2: float = 0.0
    peak_idx: Optional[int] = None
    history_start: int = 0


def classify_dialogue(
    logits: torch.Tensor,
    base_history_window: int = 4,
    max_history_window: int = 8,
    eps: float = 1e-6,
) -> list[CausalDecision]:
    """Classify every utterance using only its own and preceding cached logits.

    ``logits[t]`` is the previous epoch's original/final prediction for utterance
    ``t``. The body text of the paper gives the dimensionally consistent rule:
    compare the current logit vector with the historical mean and the Peak-End
    prototype. The first utterance has no history and keeps the standard prompt.
    """

    if logits.ndim != 2:
        raise ValueError(f"Expected [num_utterances, num_classes], got {tuple(logits.shape)}")
    if base_history_window < 1 or max_history_window < base_history_window:
        raise ValueError("Require 1 <= base_history_window <= max_history_window")

    cached = logits.detach().float()
    decisions: list[CausalDecision] = []
    for t in range(cached.size(0)):
        if t == 0:
            decisions.append(CausalDecision(None, history_start=0))
            continue

        start = max(0, t - max_history_window)
        history = cached[start:t]
        current = cached[t]
        mean_state = history.mean(dim=0)
        intensities = history.norm(p=2, dim=-1)
        peak_rel = int(intensities.argmax().item())
        peak_idx = start + peak_rel
        peak_end_state = (history[peak_rel] + history[-1]) / 2.0

        lambda_c1 = float((current - mean_state).norm(p=2).item())
        lambda_c2 = float((current - peak_end_state).norm(p=2).item())
        causal_type = 0 if lambda_c1 < lambda_c2 else 1

        if causal_type == 0:
            # Paper: expand the baseline context by lambda_C2/lambda_C1.
            ratio = lambda_c2 / max(lambda_c1, eps)
            history_len = min(max_history_window, max(base_history_window,
                              int(math.ceil(base_history_window * ratio))))
        else:
            history_len = base_history_window

        decisions.append(CausalDecision(
            causal_type=causal_type,
            lambda_c1=lambda_c1,
            lambda_c2=lambda_c2,
            peak_idx=peak_idx,
            history_start=max(0, t - history_len),
        ))
    return decisions


def _clean(text: str) -> str:
    return " ".join(str(text).replace("'", "").replace('"', "").split())


def build_erc_prompt(
    speakers: Sequence[str],
    texts: Sequence[str],
    target_idx: int,
    emotion_names: Sequence[str],
    decision: Optional[CausalDecision] = None,
    base_history_window: int = 4,
) -> str:
    """Build instruction + history + optional causal prompt + label statement."""

    if not (0 <= target_idx < len(texts)) or len(speakers) != len(texts):
        raise ValueError("Invalid target index or mismatched speakers/texts")

    if decision is None:
        history_start = max(0, target_idx - base_history_window)
        causal_text = ""
    else:
        history_start = min(target_idx, max(0, decision.history_start))
        causal_text = "" if decision.causal_type is None else (
            C1_PROMPT if decision.causal_type == 0 else C2_PROMPT
        )

    history = [
        f"{speakers[i]}: {_clean(texts[i])}"
        for i in range(history_start, target_idx)
    ]

    peak_line = ""
    if decision is not None and decision.causal_type == 1 and decision.peak_idx is not None:
        p = decision.peak_idx
        if 0 <= p < target_idx:
            peak_line = f"\nPeak utterance: {speakers[p]}: {_clean(texts[p])}"

    target = (
        f"{speakers[target_idx]}: <|audio_feat|> <|visual_feat|> "
        f"<|text_feat|> {_clean(texts[target_idx])}"
    )
    history_block = "\n".join(history + [target])
    labels = ", ".join(emotion_names)
    causal_block = f"\nCausal guidance: {causal_text}" if causal_text else ""

    return (
        "Now you are an expert in sentiment and emotional analysis. The following "
        "conversation contains multiple speakers."
        f"{causal_block}\nConversation:\n{history_block}{peak_line}\n"
        f"Please select the emotional label of <{speakers[target_idx]}: "
        f"{_clean(texts[target_idx])}> from [{labels}]."
    )
