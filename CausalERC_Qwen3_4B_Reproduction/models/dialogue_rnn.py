from __future__ import annotations
import torch
from torch import nn

class DialogueRNN(nn.Module):
    """Causal speaker/context/emotion recurrent state encoder."""
    def __init__(self, input_dim, hidden_dim=200, max_speakers=16, dropout=0.3):
        super().__init__(); self.speaker=nn.GRUCell(input_dim,hidden_dim); self.context=nn.GRUCell(hidden_dim,hidden_dim); self.emotion=nn.GRUCell(hidden_dim*2,hidden_dim); self.spk_emb=nn.Embedding(max_speakers,input_dim); self.norm=nn.LayerNorm(input_dim); self.drop=nn.Dropout(dropout)
    def forward(self,x,speaker_ids,lengths=None):
        B,T,_=x.shape; c=x.new_zeros(B,self.context.hidden_size); e=c.clone(); out=[]
        speaker_states = {}
        for t in range(T):
            active=torch.ones(B,dtype=torch.bool,device=x.device) if lengths is None else t<lengths.to(x.device)
            z=self.norm(x[:,t]); qn=[]
            for b in range(B):
                sid=int(speaker_ids[b,t]); key=(b,sid)
                previous = speaker_states.get(key, x.new_zeros(self.speaker.hidden_size))
                inp=z[b]+self.spk_emb(speaker_ids[b,t]); updated=self.speaker(self.drop(inp),previous)
                speaker_states[key]=updated; qn.append(updated)
            qn=torch.stack(qn); cn=self.context(qn,c); en=self.emotion(torch.cat([qn,cn],-1),e)
            c=torch.where(active[:,None],cn,c); e=torch.where(active[:,None],en,e); out.append(e)
        return torch.stack(out,1)
