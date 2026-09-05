"""Peak-End causal classification for the two-stage Causal-ERC baseline."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .prompt import CausalDecision

@dataclass(frozen=True)
class ReasoningResult:
    decisions: list[CausalDecision]
    causal_type: int

class CausalReasoner:
    """Classify each current utterance as C1 or C2 using Peak-End distances.

    ``initial_logits`` come from the current batch's Stage-1 forward pass. For
    utterance *t*, C1 compares the current emotion vector with the historical
    mean; C2 compares it with the mean of the historical peak and end vectors.
    No epoch cache, gold label or future utterance is consulted.
    """
    def __init__(self, history_window: int = 4):
        if history_window < 1: raise ValueError("history_window must be positive")
        self.history_window = history_window

    def decide(self, dialogue, initial_logits: torch.Tensor, emotion_context: torch.Tensor | None = None):
        if initial_logits.ndim != 2: raise ValueError("initial_logits must be [utterance, classes]")
        decisions=[]; n=initial_logits.size(0)
        for t in range(n):
            start=max(0,t-self.history_window)
            if t==0:
                decisions.append(CausalDecision(0,0,None)); continue
            history=initial_logits[start:t].float(); current=initial_logits[t].float(); mean_state=history.mean(0)
            peak_rel=int(history.norm(p=2,dim=-1).argmax()); peak_end=(history[peak_rel]+history[-1])/2
            d_c1=(current-mean_state).norm(p=2); d_c2=(current-peak_end).norm(p=2)
            causal=0 if d_c1 < d_c2 else 1
            decisions.append(CausalDecision(causal,start,start+peak_rel if causal==1 else None))
        return decisions

    def decide_batch(self, dialogues, initial_logits, emotion_context=None):
        result={}; offset=0
        for dialogue in dialogues:
            n=dialogue["utt_count"]; ctx=None if emotion_context is None else emotion_context[offset:offset+n]
            result[dialogue["vid"]]=self.decide(dialogue,initial_logits[offset:offset+n],ctx); offset+=n
        if offset!=initial_logits.size(0): raise ValueError("dialogue/logit count mismatch")
        return result
