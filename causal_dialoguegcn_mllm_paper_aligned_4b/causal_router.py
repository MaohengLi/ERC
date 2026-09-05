"""Multimodal Peak-End C1/C2 routing with the original Causal-ERC semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

C1 = 0  # language/history -> current emotion; deliberate/System 2
C2 = 1  # existing emotion -> current wording; reactive/System 1


@dataclass(frozen=True)
class DualSystemDecision:
    causal_type: Optional[int]  # None=standard, C1=0, C2=1
    lambda_c1: float = 0.0
    lambda_c2: float = 0.0
    peak_idx: Optional[int] = None
    history_start: int = 0
    alpha_system2: float = 0.5
    reliability: float = 0.5
    disagreement: float = 0.0
    peak_intensity: float = 0.0

    @property
    def route_name(self) -> str:
        if self.causal_type is None:
            return "standard"
        return "system2_c1" if self.causal_type == C1 else "system1_c2"


def _reliability(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    entropy = -(probabilities.clamp_min(1e-8) *
                probabilities.clamp_min(1e-8).log()).sum(-1)
    return 1.0 - entropy / math.log(logits.size(-1))


def classify_multimodal_dialogue(
    route_logits: torch.Tensor,
    base_history_window: int = 4,
    max_history_window: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
) -> list[DualSystemDecision]:
    """Return decisions from previous-pass [T, A, V, fused] logits.

    The fused fourth branch is the LLM's ordinary Causal-ERC prediction.  The
    first three branches expose audio/video/text evidence to Peak-End without
    adding any extra token to the LLM interface.  Only utterances preceding the
    target are inspected, exactly as in the paper's causal prompting loop.
    """
    if route_logits.ndim != 3 or route_logits.size(1) != 4:
        raise ValueError("Expected route_logits with shape [N, 4, C] = T/A/V/fused")
    if base_history_window < 1 or max_history_window < base_history_window:
        raise ValueError("Require 1 <= base_history_window <= max_history_window")

    logits = route_logits.detach().float()
    probabilities = F.softmax(logits, dim=-1)
    decisions: list[DualSystemDecision] = []
    for t in range(logits.size(0)):
        if t == 0:
            decisions.append(DualSystemDecision(None))
            continue
        start = max(0, t - max_history_window)
        history_logits = logits[start:t]
        current = logits[t]
        history_prob = probabilities[start:t]
        mean_state = history_logits.mean(dim=0)
        intensity = history_logits.norm(p=2, dim=-1).mean(dim=1)
        peak_rel = int(intensity.argmax().item())
        peak_idx = start + peak_rel
        peak_end = (history_logits[peak_rel] + history_logits[-1]) / 2.0

        rel = _reliability(current)
        d_c1 = (current - mean_state).norm(p=2, dim=-1)
        d_c2 = (current - peak_end).norm(p=2, dim=-1)
        rel_sum = rel.sum().clamp_min(eps)
        lambda_c1 = float((d_c1 * rel).sum().div(rel_sum).item())
        lambda_c2 = float((d_c2 * rel).sum().div(rel_sum).item())
        causal_type = C1 if lambda_c1 < lambda_c2 else C2
        alpha = float(torch.sigmoid(torch.tensor(
            (lambda_c2 - lambda_c1) / max(temperature, eps))).item())
        if causal_type == C1:
            ratio = lambda_c2 / max(lambda_c1, eps)
            history_len = min(max_history_window,
                              max(base_history_window,
                                  int(math.ceil(base_history_window * ratio))))
        else:
            history_len = base_history_window
        decisions.append(DualSystemDecision(
            causal_type=causal_type, lambda_c1=lambda_c1, lambda_c2=lambda_c2,
            peak_idx=peak_idx, history_start=max(0, t - history_len),
            alpha_system2=alpha, reliability=float(rel.mean().item()),
            disagreement=float((d_c1 - d_c2).abs().mean().item()),
            peak_intensity=float(intensity[peak_rel].item()),
        ))
    return decisions


def decisions_to_gate_features(
    decisions: Optional[Sequence[DualSystemDecision]], count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return non-LLM route features used only by the pre-projector gate."""
    if decisions is None:
        values = [[0.5, 0.5, 0.0, 0.0, 0.5] for _ in range(count)]
    else:
        if len(decisions) != count:
            raise ValueError(f"Expected {count} decisions, got {len(decisions)}")
        values = [[d.alpha_system2, d.reliability, d.disagreement,
                   d.peak_intensity, 1.0 - d.alpha_system2] for d in decisions]
    return torch.tensor(values, dtype=torch.float32, device=device)


__all__ = ["C1", "C2", "DualSystemDecision", "classify_multimodal_dialogue",
           "decisions_to_gate_features"]

