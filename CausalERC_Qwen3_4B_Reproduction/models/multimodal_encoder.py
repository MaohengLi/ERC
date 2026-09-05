from __future__ import annotations
import torch
from torch import nn
from .dialogue_rnn import DialogueRNN

class ModalityEncoder(nn.Module):
    def __init__(self,in_dim,hidden_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.LayerNorm(hidden_dim))
    def forward(self,x): return self.net(x)

class MultimodalEncoder(nn.Module):
    def __init__(self,text_dim,audio_dim,visual_dim,hidden_dim=200,heads=4,dropout=0.3):
        super().__init__(); self.text=ModalityEncoder(text_dim,hidden_dim); self.audio=ModalityEncoder(audio_dim,hidden_dim); self.visual=ModalityEncoder(visual_dim,hidden_dim); self.rnn_t=DialogueRNN(hidden_dim,hidden_dim,dropout=dropout); self.rnn_a=DialogueRNN(hidden_dim,hidden_dim,dropout=dropout); self.rnn_v=DialogueRNN(hidden_dim,hidden_dim,dropout=dropout); self.cross_attention=nn.MultiheadAttention(hidden_dim,heads,batch_first=True,dropout=dropout); self.norm=nn.LayerNorm(hidden_dim)
    def forward(self,text,audio,visual,speaker_ids,lengths=None):
        t=self.rnn_t(self.text(text),speaker_ids,lengths); a=self.rnn_a(self.audio(audio),speaker_ids,lengths); v=self.rnn_v(self.visual(visual),speaker_ids,lengths)
        tokens=torch.stack([t,a,v],dim=2); flat=tokens.reshape(-1,3,tokens.size(-1)); attended,_=self.cross_attention(flat,flat,flat); attended=self.norm(flat+attended).reshape_as(tokens)
        return {"text":attended[:,:,0],"audio":attended[:,:,1],"visual":attended[:,:,2],"multimodal":attended.mean(dim=2)}
