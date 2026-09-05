"""Multimodal Peak-End memory and differentiable System-1/System-2 routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

C1 = 0  # language/history causes the current emotion (deliberative/System 2)
C2 = 1  # existing emotion causes the current wording (reactive/System 1)


@dataclass(frozen=True)
class DualSystemDecision:
    """A decision for one utterance, computed without gradients or future labels."""

    causal_type: Optional[int]  # None=standard, C1=0, C2=1
    alpha_system2: float
    lambda_c1: float
    lambda_c2: float
    reliability: float
    disagreement: float
    peak_idx: Optional[int]
    history_start: int
    peak_intensity: float

    @property
    def route_name(self) -> str:
        if self.causal_type is None:
            return "standard"
        return "system2_c1" if self.causal_type == 0 else "system1_c2"


def _reliability(probabilities: torch.Tensor) -> torch.Tensor:
    """1 - normalised entropy, in [0, 1], for [modalities, classes]."""

    classes = probabilities.size(-1)
    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(-1)
    return 1.0 - entropy / math.log(classes)


def classify_multimodal_dialogue(
    modality_logits: torch.Tensor,
    base_history_window: int = 4,
    max_history_window: int = 8,
    temperature: float = 0.5,
    neutral_index: int = 2,
    eps: float = 1e-6,
) -> list[DualSystemDecision]:
    """Classify a dialogue from cached previous-epoch T/A/V/Graph logits.

    Args:
        modality_logits: ``[num_utterances, 4, num_classes]``. The four
            modalities are text, audio, visual and graph. Only rows before the
            current utterance are used to choose the peak/end context.

    The distances are measured in a shared probability simplex rather than raw
    embedding norms, making audio/video/text scales comparable. Reliability
    weights down-weight uncertain modalities; the returned ``alpha_system2`` is
    a soft route used both by the model and the prompt composer.
    """

    if modality_logits.ndim != 3 or modality_logits.size(1) != 4:
        raise ValueError(
            "Expected modality_logits [N, 4, C] for text/audio/visual/graph"
        )
    if base_history_window < 1 or max_history_window < base_history_window:
        raise ValueError("Require 1 <= base_history_window <= max_history_window")
    if not 0 <= neutral_index < modality_logits.size(-1):
        raise ValueError("neutral_index is outside the class dimension")

    probs = F.softmax(modality_logits.detach().float(), dim=-1)
    decisions: list[DualSystemDecision] = []
    for t in range(probs.size(0)):
        if t == 0:
            decisions.append(DualSystemDecision(
                causal_type=None, alpha_system2=0.5,
                lambda_c1=0.0, lambda_c2=0.0, reliability=0.5,
                disagreement=0.0, peak_idx=None, history_start=0,
                peak_intensity=0.0,
            ))
            continue

        start = max(0, t - max_history_window)
        history = probs[start:t]  # [H, 4, C]
        current = probs[t]
        mean_state = history.mean(dim=0)
        # Peak is emotional intensity, not merely classifier confidence.  For
        # IEMOCAP this is non-neutral probability, reliability-weighted and
        # averaged over T/A/V/G. Thus audio/video peaks explicitly participate.
        emotional = 1.0 - history[..., neutral_index]
        history_reliability = _reliability(history)
        intensities = (emotional * (0.5 + 0.5 * history_reliability)).mean(dim=1)
        peak_rel = int(intensities.argmax().item())
        peak_idx = start + peak_rel
        peak_end = (history[peak_rel] + history[-1]) / 2.0

        d_c1_modal = (current - mean_state).norm(p=2, dim=-1)
        d_c2_modal = (current - peak_end).norm(p=2, dim=-1)
        rel_modal = _reliability(current)
        rel_sum = rel_modal.sum().clamp_min(eps)
        lambda_c1 = float((d_c1_modal * rel_modal).sum().div(rel_sum).item())
        lambda_c2 = float((d_c2_modal * rel_modal).sum().div(rel_sum).item())
        reliability = float(rel_modal.mean().item())
        disagreement = float((d_c1_modal - d_c2_modal).abs().mean().item())

        # Exact Causal-ERC semantics: C1 is language/history -> emotion and is
        # selected when the history-mean distance is smaller. C2 is emotion ->
        # wording and is selected when peak/end is closer. Soft alpha remains
        # available at the model interface although the cached type is hard.
        causal_type = C1 if lambda_c1 < lambda_c2 else C2
        alpha_system2 = float(torch.sigmoid(
            torch.tensor((lambda_c2 - lambda_c1) / max(temperature, eps))
        ).item())
        if causal_type == C1:
            ratio = lambda_c2 / max(lambda_c1, eps)
            history_len = min(
                max_history_window,
                max(base_history_window, int(math.ceil(base_history_window * ratio))),
            )
        else:
            history_len = base_history_window

        decisions.append(DualSystemDecision(
            causal_type=causal_type,
            alpha_system2=alpha_system2,
            lambda_c1=lambda_c1,
            lambda_c2=lambda_c2,
            reliability=reliability,
            disagreement=disagreement,
            peak_idx=peak_idx,
            history_start=max(0, t - history_len),
            peak_intensity=float(intensities[peak_rel].item()),
        ))
    return decisions


def decisions_to_features(
    decisions: Optional[Sequence[DualSystemDecision]],
    count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert decisions to the 5-D route token input used by the model."""

    if decisions is None:
        values = [[0.5, 0.5, 0.0, 0.0, 0.5] for _ in range(count)]
    else:
        if len(decisions) != count:
            raise ValueError(f"Expected {count} decisions, got {len(decisions)}")
        values = [
            [
                d.alpha_system2,
                d.reliability,
                d.disagreement,
                d.peak_intensity,
                1.0 - d.alpha_system2,
            ]
            for d in decisions
        ]
    return torch.tensor(values, dtype=torch.float32, device=device)
