"""Independent modality projections, DialogueGCNs and cross-modal fusion."""
from __future__ import annotations
import torch
from torch import nn
from .dialogue_gcn import DialogueGCN
from .graph_builder import DialogueGraphBuilder

class ModalityEncoder(nn.Module):
    def __init__(self,in_dim,hidden_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LayerNorm(hidden_dim))
    def forward(self,x): return self.net(x)

class MultimodalEncoder(nn.Module):
    def __init__(self,text_dim,audio_dim,visual_dim,hidden_dim=200,heads=4,dropout=0.3,gcn_layers=2,context_window=10):
        super().__init__(); self.text=ModalityEncoder(text_dim,hidden_dim); self.audio=ModalityEncoder(audio_dim,hidden_dim); self.visual=ModalityEncoder(visual_dim,hidden_dim); self.graph_builder=DialogueGraphBuilder(context_window); self.gcn_t=DialogueGCN(hidden_dim,gcn_layers,dropout); self.gcn_a=DialogueGCN(hidden_dim,gcn_layers,dropout); self.gcn_v=DialogueGCN(hidden_dim,gcn_layers,dropout); self.cross_attention=nn.MultiheadAttention(hidden_dim,heads,batch_first=True,dropout=dropout); self.norm=nn.LayerNorm(hidden_dim)

    @staticmethod
    def _flatten(x,lengths): return torch.cat([x[b,:int(lengths[b])] for b in range(x.size(0))])
    @staticmethod
    def _pad(x,lengths,max_steps):
        out=x.new_zeros(lengths.numel(),max_steps,x.size(-1)); offset=0
        for b,n in enumerate(lengths.tolist()): out[b,:n]=x[offset:offset+n]; offset+=n
        return out

    def forward(self,text,audio,visual,speaker_ids,lengths):
        graph=self.graph_builder(speaker_ids,lengths); t=self.gcn_t(self._flatten(self.text(text),lengths),graph); a=self.gcn_a(self._flatten(self.audio(audio),lengths),graph); v=self.gcn_v(self._flatten(self.visual(visual),lengths),graph)
        tokens=torch.stack([t,a,v],dim=1); attended,_=self.cross_attention(tokens,tokens,tokens); attended=self.norm(tokens+attended)
        T=text.size(1); return {"text":self._pad(attended[:,0],lengths,T),"audio":self._pad(attended[:,1],lengths,T),"visual":self._pad(attended[:,2],lengths,T),"multimodal":self._pad(attended.mean(1),lengths,T),"graph":graph}
