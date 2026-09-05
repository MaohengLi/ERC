from dataclasses import dataclass
from typing import Optional
import torch
C1="Historical dialogue context causes the current emotion; analyze speaker-aware history."
C2="An existing emotional state causes the current wording; focus on peak and recent utterance."
@dataclass(frozen=True)
class CausalDecision:
    causal_type: Optional[int]=None
    peak_idx: Optional[int]=None
    history_start: int=0
def classify_dialogue(logits,history_window=4):
    out=[]
    for t in range(logits.size(0)):
        if t==0: out.append(CausalDecision()); continue
        s=max(0,t-history_window); h=logits[s:t].float(); cur=logits[t].float(); peak=s+int(h.norm(dim=-1).argmax()); c1=(cur-h.mean(0)).norm(); c2=(cur-(h[peak-s]+h[-1])/2).norm(); out.append(CausalDecision(0 if c1<c2 else 1,peak,s))
    return out
def build_prompt(speakers,texts,target_idx,labels,history_window=4,decision=None):
    if len(speakers)!=len(texts) or not 0<=target_idx<len(texts): raise ValueError("invalid dialogue")
    s=max(0,target_idx-history_window) if decision is None else min(target_idx,decision.history_start); guide="" if decision is None or decision.causal_type is None else (C1 if decision.causal_type==0 else C2); lines=[f"{speakers[i]}: {texts[i]}" for i in range(s,target_idx)]; lines.append(f"{speakers[target_idx]}: <|audio_feat|> <|visual_feat|> <|text_feat|> {texts[target_idx]}")
    if decision is not None and decision.causal_type==1 and decision.peak_idx is not None and decision.peak_idx<target_idx: lines.append(f"Peak utterance: {speakers[decision.peak_idx]}: {texts[decision.peak_idx]}")
    return "You are an expert in conversational emotion recognition."+(f"\nCausal guidance: {guide}" if guide else "")+"\nConversation:\n"+"\n".join(lines)+f"\nSelect one label from [{', '.join(labels)}]."
