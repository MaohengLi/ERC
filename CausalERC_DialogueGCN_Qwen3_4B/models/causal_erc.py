"""Causal-ERC with DialogueGCN and model-owned two-stage reasoning."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .multimodal_encoder import MultimodalEncoder
from .qwen_wrapper import Qwen3Wrapper
from .soft_hgr import soft_hgr_loss
from utils.causal_reasoner import CausalReasoner

class CausalERCDialogueGCNQwen3(nn.Module):
    def __init__(self,llm,token_ids,text_dim,audio_dim,visual_dim,hidden_dim=200,num_classes=6,heads=4,dropout=0.3,lambda_hgr=0.1,gcn_layers=2,context_window=10,history_window=4,llm_micro_batch=1):
        super().__init__(); self.lambda_hgr=lambda_hgr; self.token_ids=token_ids; self.encoder=MultimodalEncoder(text_dim,audio_dim,visual_dim,hidden_dim,heads,dropout,gcn_layers,context_window); self.qwen=Qwen3Wrapper(llm,token_ids,hidden_dim,num_classes,llm_micro_batch); self.causal_reasoner=CausalReasoner(history_window)

    def encode_context(self,batch):
        f=self.encoder(batch["text"],batch["audio"],batch["visual"],batch["speaker_ids"],batch["lengths"]); lengths=batch["lengths"]; flat=lambda x:torch.cat([x[b,:int(lengths[b])] for b in range(x.size(0))]); return f,flat(f["text"]),flat(f["audio"]),flat(f["visual"])

    def _run_qwen(self,batch,t,a,v): return self.qwen(batch["input_ids"],batch["attention_mask"],a,v,t,(batch["audio_positions"],batch["visual_positions"],batch["text_positions"]))

    @torch.no_grad()
    def initial_forward(self,batch):
        f,t,a,v=self.encode_context(batch); return {"initial_logits":self._run_qwen(batch,t,a,v),"context":f,"text_context":t,"audio_context":a,"visual_context":v}

    def _single_forward(self,batch,labels=None):
        f,t,a,v=self.encode_context(batch); logits=self._run_qwen(batch,t,a,v); hgr=soft_hgr_loss(t,a,v); ce=None if labels is None else F.cross_entropy(logits,labels); return {"logits":logits,"loss_ce":ce,"loss_hgr":hgr,"loss":None if ce is None else ce+self.lambda_hgr*hgr,"context":f}

    def forward(self,batch,labels=None,tokenizer=None,dialogues=None,max_length=768,history_window=4):
        if tokenizer is None or dialogues is None: return self._single_forward(batch,labels)
        first=self.initial_forward(batch); lengths=batch["lengths"]; emotion=torch.cat([first["context"]["multimodal"][b,:int(lengths[b])] for b in range(len(dialogues))]); decisions=self.causal_reasoner.decide_batch(dialogues,first["initial_logits"],emotion)
        from data.dataset import collate_dialogues
        final=collate_dialogues(dialogues,tokenizer,self.token_ids,max_length,history_window,decisions=decisions); final={k:(v.to(batch["input_ids"].device) if isinstance(v,torch.Tensor) else v) for k,v in final.items()}; result=self._single_forward(final,final["labels"] if labels is None else labels); result["initial_logits"]=first["initial_logits"]; result["causal_decisions"]=decisions; return result

    def parameter_report(self): return {"total":sum(p.numel() for p in self.parameters()),"trainable":sum(p.numel() for p in self.parameters() if p.requires_grad)}

CausalERC_DialogueGCN_Qwen3 = CausalERCDialogueGCNQwen3
CausalERC_DialogueGCN_Qwen3_4B = CausalERCDialogueGCNQwen3
