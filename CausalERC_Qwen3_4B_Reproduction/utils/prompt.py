"""Single-pass Causal-ERC prompt construction."""
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
C1="Historical dialogue context causes the current emotion. Analyze previous dialogue information and emotional triggers."
C2="Existing emotional state causes current wording. Analyze emotional evolution and emotional expression."
@dataclass(frozen=True)
class CausalDecision:
    causal_type:int
    history_start:int
    peak_idx:Optional[int]=None
def select_causal_type(speakers:Sequence[str],texts:Sequence[str],target_idx:int,emotion_context:Optional[torch.Tensor]=None):
    start=max(0,target_idx-4)
    if emotion_context is not None and emotion_context.numel():
        z=emotion_context.float().reshape(-1,emotion_context.shape[-1]); delta=float((z[-1]-z[:-1].mean(0)).norm()) if z.size(0)>1 else 0.; causal=1 if delta>=float(z.norm(dim=-1).mean()) else 0
    else:
        current=str(texts[target_idx]).lower(); causal=1 if any(k in current for k in ("!","?","sorry","hate","love","can't","cannot")) else 0
    return CausalDecision(causal,start,start if causal==1 and target_idx>start else None)
def build_prompt(speakers:Sequence[str],texts:Sequence[str],target_idx:int,labels:Sequence[str],history_window:int=4,decision:Optional[CausalDecision]=None,standard:bool=False):
    if len(speakers)!=len(texts) or not 0<=target_idx<len(texts): raise ValueError("invalid dialogue")
    d=decision or select_causal_type(speakers,texts,target_idx); start=max(0,target_idx-history_window) if decision is None else min(target_idx,d.history_start); guide=C1 if d.causal_type==0 else C2; lines=[f"{speakers[i]}: {texts[i]}" for i in range(start,target_idx)]; lines.append(f"{speakers[target_idx]}: <|audio_feat|> <|visual_feat|> <|text_feat|> {texts[target_idx]}")
    if d.causal_type==1 and d.peak_idx is not None and d.peak_idx<target_idx: lines.append(f"Peak context: {speakers[d.peak_idx]}: {texts[d.peak_idx]}")
    guidance="" if standard else "\nCausal guidance: "+guide
    return "You are an expert in conversational emotion recognition."+guidance+"\nConversation:\n"+"\n".join(lines)+f"\nSelect one label from [{', '.join(labels)}]."
