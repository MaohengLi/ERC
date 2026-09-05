from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .multimodal_encoder import MultimodalEncoder
from .qwen_wrapper import Qwen3Wrapper
from .soft_hgr import soft_hgr_loss

class CausalERCQwen3(nn.Module):
    def __init__(self,llm,token_ids,text_dim,audio_dim,visual_dim,hidden_dim=200,num_classes=6,heads=4,dropout=0.3,lambda_hgr=0.1):
        super().__init__(); self.lambda_hgr=lambda_hgr; self.encoder=MultimodalEncoder(text_dim,audio_dim,visual_dim,hidden_dim,heads,dropout); self.qwen=Qwen3Wrapper(llm,token_ids,hidden_dim,num_classes)
    def forward(self,batch,labels=None):
        f=self.encoder(batch["text"],batch["audio"],batch["visual"],batch["speaker_ids"],batch.get("lengths")); lengths=batch["lengths"].to(batch["input_ids"].device); flat=lambda x:torch.cat([x[b,:int(lengths[b])] for b in range(x.size(0))]); t,a,v=flat(f["text"]),flat(f["audio"]),flat(f["visual"]); logits=self.qwen(batch["input_ids"],batch["attention_mask"],a,v,t,(batch["audio_positions"],batch["visual_positions"],batch["text_positions"])); h=soft_hgr_loss(t,a,v); ce=None if labels is None else F.cross_entropy(logits,labels); return {"logits":logits,"loss_ce":ce,"loss_hgr":h,"loss":None if ce is None else ce+self.lambda_hgr*h}
    def parameter_report(self):
        return {"total":sum(p.numel() for p in self.parameters()),"trainable":sum(p.numel() for p in self.parameters() if p.requires_grad)}
