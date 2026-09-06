"""Bidirectional dialogue graph construction for offline conversational ERC."""
from __future__ import annotations
from dataclasses import dataclass
import torch

SELF = 0
SAME_SPEAKER = 1
INTER_SPEAKER = 2
TEMPORAL = 3
NUM_RELATIONS = 4

@dataclass
class DialogueGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int
    def to(self, device):
        return DialogueGraph(self.edge_index.to(device), self.edge_type.to(device), self.num_nodes)

class DialogueGraphBuilder:
    """Build a local bidirectional graph over each complete dialogue for offline ERC."""
    def __init__(self, context_window: int = 10):
        if context_window < 1: raise ValueError("context_window must be positive")
        self.context_window = context_window

    def __call__(self, speaker_ids: torch.Tensor, lengths: torch.Tensor) -> DialogueGraph:
        sources, targets, relations = [], [], []; offset = 0
        for b in range(speaker_ids.size(0)):
            n = int(lengths[b]); speakers = speaker_ids[b, :n].tolist()
            for i in range(n):
                sources.append(offset+i); targets.append(offset+i); relations.append(SELF)
            for i in range(n):
                for j in range(i+1, min(n, i+self.context_window+1)):
                    rel = SAME_SPEAKER if speakers[i] == speakers[j] else INTER_SPEAKER
                    sources.extend((offset+i, offset+j)); targets.extend((offset+j, offset+i)); relations.extend((rel, rel))
            for i in range(n-1):
                sources.extend((offset+i, offset+i+1)); targets.extend((offset+i+1, offset+i)); relations.extend((TEMPORAL, TEMPORAL))
            offset += n
        if sources:
            edge_index = torch.tensor([sources, targets], dtype=torch.long, device=speaker_ids.device)
            edge_type = torch.tensor(relations, dtype=torch.long, device=speaker_ids.device)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=speaker_ids.device); edge_type = torch.empty((0,), dtype=torch.long, device=speaker_ids.device)
        return DialogueGraph(edge_index, edge_type, offset)
