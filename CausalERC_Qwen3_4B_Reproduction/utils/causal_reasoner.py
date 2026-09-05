"""Batch-local causal reasoner for the two-stage Causal-ERC baseline."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .prompt import CausalDecision

@dataclass(frozen=True)
class ReasoningResult:
    decisions: list[CausalDecision]
    causal_type: int

class CausalReasoner:
    """Select C1/C2 from current history, recurrent context and stage-1 logits."""
    def decide(self, dialogue, initial_logits: torch.Tensor, emotion_context: torch.Tensor) -> list[CausalDecision]:
        decisions=[]; offset=0
        for idx in range(dialogue["utt_count"]):
            logits=initial_logits[offset+idx]
            ctx=emotion_context[offset:offset+idx+1]
            if idx==0: causal=0
            else:
                state_delta=(ctx[-1]-ctx[:-1].mean(0)).norm()
                logit_delta=(logits-initial_logits[offset:offset+idx].mean(0)).norm()
                causal=1 if float(state_delta+logit_delta) >= float(ctx.norm(dim=-1).mean()) else 0
            start=max(0,idx-4)
            decisions.append(CausalDecision(causal, start, start if causal==1 and idx>start else None))
        return decisions

    def decide_batch(self, dialogues, initial_logits, emotion_context):
        result={}; offset=0
        for dialogue in dialogues:
            n=dialogue["utt_count"]; result[dialogue["vid"]]=self.decide(dialogue,initial_logits[offset:offset+n],emotion_context[offset:offset+n]); offset+=n
        if offset!=initial_logits.size(0): raise ValueError("dialogue/logit count mismatch")
        return result
