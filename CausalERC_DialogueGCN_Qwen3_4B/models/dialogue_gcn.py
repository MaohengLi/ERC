"""Relation-specific DialogueGCN implemented without external graph packages."""
from __future__ import annotations
import torch
from torch import nn
from .graph_builder import DialogueGraph, NUM_RELATIONS

class RelationalGraphConv(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int = NUM_RELATIONS, dropout: float = 0.3):
        super().__init__(); self.num_relations=num_relations; self.weight=nn.Parameter(torch.empty(num_relations,hidden_dim,hidden_dim)); self.bias=nn.Parameter(torch.zeros(hidden_dim)); self.norm=nn.LayerNorm(hidden_dim); self.dropout=nn.Dropout(dropout); nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes: torch.Tensor, graph: DialogueGraph) -> torch.Tensor:
        if graph.num_nodes != nodes.size(0): raise ValueError("graph/node count mismatch")
        if graph.edge_type.numel()==0: return self.norm(nodes)
        src,dst=graph.edge_index; rel=graph.edge_type
        messages=torch.bmm(nodes[src].unsqueeze(1),self.weight[rel]).squeeze(1)
        keys=dst*self.num_relations+rel; degree=torch.bincount(keys,minlength=nodes.size(0)*self.num_relations).clamp_min(1).to(nodes.dtype)
        alpha=degree[keys].reciprocal(); aggregate=torch.zeros_like(nodes); aggregate.index_add_(0,dst,messages*alpha.unsqueeze(-1))
        return self.norm(nodes+self.dropout(torch.relu(aggregate+self.bias)))

class DialogueGCN(nn.Module):
    def __init__(self, hidden_dim: int = 200, layers: int = 2, dropout: float = 0.3):
        super().__init__(); self.layers=nn.ModuleList([RelationalGraphConv(hidden_dim,dropout=dropout) for _ in range(layers)])
    def forward(self,nodes,graph):
        for layer in self.layers: nodes=layer(nodes,graph)
        return nodes
