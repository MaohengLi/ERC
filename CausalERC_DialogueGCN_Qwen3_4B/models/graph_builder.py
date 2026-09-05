"""Causal multi-relational dialogue graph construction."""
from __future__ import annotations
from dataclasses import dataclass
import torch

SAME_SPEAKER = 0
INTER_SPEAKER = 1
TEMPORAL = 2
NUM_RELATIONS = 3

@dataclass
class DialogueGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int

    def to(self, device):
        return DialogueGraph(self.edge_index.to(device), self.edge_type.to(device), self.num_nodes)

class DialogueGraphBuilder:
    """Build past-to-current edges; future utterances never become parents."""
    def __init__(self, context_window: int = 10):
        if context_window < 1: raise ValueError("context_window must be positive")
        self.context_window = context_window

    def __call__(self, speaker_ids: torch.Tensor, lengths: torch.Tensor) -> DialogueGraph:
        sources=[]; targets=[]; relations=[]; offset=0
        for b in range(speaker_ids.size(0)):
            n=int(lengths[b]); speakers=speaker_ids[b,:n].tolist()
            for dst in range(n):
                start=max(0,dst-self.context_window)
                for src in range(start,dst):
                    sources.append(offset+src); targets.append(offset+dst)
                    relations.append(SAME_SPEAKER if speakers[src]==speakers[dst] else INTER_SPEAKER)
                if dst>0:
                    sources.append(offset+dst-1); targets.append(offset+dst); relations.append(TEMPORAL)
            offset+=n
        if sources:
            edge_index=torch.tensor([sources,targets],dtype=torch.long,device=speaker_ids.device)
            edge_type=torch.tensor(relations,dtype=torch.long,device=speaker_ids.device)
        else:
            edge_index=torch.empty((2,0),dtype=torch.long,device=speaker_ids.device)
            edge_type=torch.empty((0,),dtype=torch.long,device=speaker_ids.device)
        return DialogueGraph(edge_index,edge_type,offset)
