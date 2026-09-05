"""Causal DialogueGCN-LLM with multimodal dual-system routing.

The implementation deliberately keeps the graph encoder dependency-free.  A
dense, local temporal graph is evaluated dialogue by dialogue, so utterances
from two different conversations can never exchange messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def soft_hgr_loss(features: Sequence[Tensor], eps: float = 1e-6) -> Tensor:
    """A stable Soft-HGR-style dependence penalty.

    HGR maximises nonlinear dependence between views.  For the auxiliary
    multimodal branch we use the common finite-sample relaxation: centre each
    view, whiten its coordinates by the sample standard deviation, and penalise
    squared cross-view correlations.  It is bounded, differentiable and works
    even when a batch contains a single dialogue.
    """

    if len(features) < 2:
        return features[0].new_zeros(())
    losses = []
    for i in range(len(features)):
        xi = features[i]
        xi = xi - xi.mean(dim=0, keepdim=True)
        xi = xi / (xi.std(dim=0, unbiased=False, keepdim=True) + eps)
        for j in range(i + 1, len(features)):
            xj = features[j]
            xj = xj - xj.mean(dim=0, keepdim=True)
            xj = xj / (xj.std(dim=0, unbiased=False, keepdim=True) + eps)
            # Coordinates are aligned by the shared cross-modal attention.
            # Minimising 1-rho^2 maximises nonlinear dependence without the
            # unbounded scale failure of raw covariance objectives.
            corr = (xi * xj).mean(dim=0).clamp(-1.0, 1.0)
            losses.append(1.0 - corr.square().mean())
    return torch.stack(losses).mean()


class DialogueGCNLayer(nn.Module):
    """Local relational graph attention for one dialogue.

    Relation types encode source/target speaker identity and temporal
    direction.  The bounded temporal window avoids the quadratic memory cost
    of a fully connected graph while preserving DialogueGCN's speaker-aware
    context propagation.
    """

    def __init__(self, hidden_dim: int, num_relations: int = 8,
                 window: int = 10, dropout: float = 0.2,
                 causal_only: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window = window
        self.causal_only = causal_only
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_weight = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.relation_weight)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _relation(src_speaker: int, dst_speaker: int, future: bool) -> int:
        # Two-speaker IEMOCAP relation inventory: source x target x direction.
        return ((int(src_speaker) % 2) * 2 + (int(dst_speaker) % 2)) * 2 + int(future)

    def forward(self, x: Tensor, speaker_ids: Tensor) -> Tensor:
        n = x.shape[0]
        if n == 0:
            return x
        q, k = self.q(x), self.k(x)
        messages = []
        scale = self.hidden_dim ** -0.5
        speakers = speaker_ids.detach().cpu().tolist()
        for i in range(n):
            lo = max(0, i - self.window)
            hi = i + 1 if self.causal_only else min(n, i + self.window + 1)
            neighbours = list(range(lo, hi))
            scores = torch.stack([torch.dot(q[i], k[j]) * scale for j in neighbours])
            weights = torch.softmax(scores, dim=0)
            agg = x.new_zeros(self.hidden_dim)
            for weight, j in zip(weights, neighbours):
                rel = self._relation(speakers[j], speakers[i], j > i)
                agg = agg + weight * torch.matmul(self.relation_weight[rel], x[j])
            messages.append(agg)
        message = torch.stack(messages, dim=0)
        y = self.out_norm(x + self.dropout(message))
        return self.ffn_norm(y + self.ffn(y))


class DialogueGCNEncoder(nn.Module):
    """Three-modality encoder with cross-attention followed by DialogueGCN."""

    def __init__(self, text_dim: int, audio_dim: int, visual_dim: int,
                 hidden_dim: int = 200, num_heads: int = 4, layers: int = 2,
                 graph_window: int = 10, dropout: float = 0.2,
                 causal_graph: bool = True) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.visual_proj = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.node_fuse = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                                       nn.GELU(), nn.LayerNorm(hidden_dim))
        self.graph_layers = nn.ModuleList([
            DialogueGCNLayer(hidden_dim, window=graph_window, dropout=dropout,
                             causal_only=causal_graph)
            for _ in range(layers)
        ])

    def forward(self, text: Tensor, audio: Tensor, visual: Tensor,
                speaker_ids: Tensor, lengths: Tensor) -> Dict[str, Tensor | list[int]]:
        # Padded positions are never used by the graph; cross-attention is only
        # across the three modality tokens of each utterance.
        ft = self.text_proj(text)
        fa = self.audio_proj(audio)
        fv = self.visual_proj(visual)
        bsz, max_len, _ = ft.shape
        modal = torch.stack((ft, fa, fv), dim=2).reshape(bsz * max_len, 3, self.hidden_dim)
        attended, _ = self.cross_attn(modal, modal, modal, need_weights=False)
        attended = self.cross_norm(modal + attended).reshape(bsz, max_len, 3, self.hidden_dim)
        ft, fa, fv = attended.unbind(dim=2)

        flat_t, flat_a, flat_v, flat_g = [], [], [], []
        counts: list[int] = []
        for b in range(bsz):
            n = int(lengths[b].item())
            counts.append(n)
            t_b, a_b, v_b = ft[b, :n], fa[b, :n], fv[b, :n]
            graph = self.node_fuse(torch.cat((t_b, a_b, v_b), dim=-1))
            spk = speaker_ids[b, :n]
            for layer in self.graph_layers:
                graph = layer(graph, spk)
            flat_t.append(t_b); flat_a.append(a_b); flat_v.append(v_b); flat_g.append(graph)
        return {
            "text": torch.cat(flat_t, dim=0), "audio": torch.cat(flat_a, dim=0),
            "visual": torch.cat(flat_v, dim=0), "graph": torch.cat(flat_g, dim=0),
            "counts": counts,
        }


class CausalDialogueGCNLLM(nn.Module):
    """Qwen-compatible 4B backbone with trainable multimodal/route adapters."""

    def __init__(self, llm: nn.Module, token_ids: Dict[str, int],
                 text_dim: int = 100, audio_dim: int = 100, visual_dim: int = 512,
                 num_classes: int = 6, hidden_dim: int = 200, num_heads: int = 4,
                 graph_layers: int = 2, graph_window: int = 10, dropout: float = 0.2,
                 lambda_hgr: float = 1.0, lambda_aux: float = 0.2,
                 llm_micro_batch: int = 4, causal_graph: bool = True) -> None:
        super().__init__()
        self.llm = llm
        self.token_ids = token_ids
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lambda_hgr = lambda_hgr
        self.lambda_aux = lambda_aux
        self.llm_micro_batch = max(1, int(llm_micro_batch))
        self.encoder = DialogueGCNEncoder(
            text_dim, audio_dim, visual_dim, hidden_dim, num_heads,
            graph_layers, graph_window, dropout, causal_graph,
        )
        self.aux_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(4)])
        llm_hidden = int(getattr(llm.config, "hidden_size", llm.config.text_config.hidden_size
                                 if hasattr(llm.config, "text_config") else hidden_dim))
        self.llm_hidden = llm_hidden
        self.modality_to_llm = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, llm_hidden))
            for name in ("audio", "visual", "text", "graph")
        })
        self.route_to_llm = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, llm_hidden), nn.Tanh())
        self.feature_scale = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Sequential(nn.LayerNorm(llm_hidden), nn.Linear(llm_hidden, num_classes))

    def parameter_report(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "llm_trainable": llm_trainable,
                "adapter_trainable": trainable - llm_trainable}

    def save_non_llm(self, path: str | Path) -> None:
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()
                 if not k.startswith("llm.")}
        torch.save(state, Path(path))

    def load_non_llm(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("llm.")]
        missing = [k for k in missing if not k.startswith("llm.")]
        if missing or unexpected:
            raise RuntimeError(f"Invalid non-LLM checkpoint: missing={missing}, unexpected={unexpected}")

    def encode_modalities(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor | list[int]]:
        return self.encoder(batch["text"], batch["audio"], batch["visual"],
                             batch["speaker_ids"], batch["lengths"])

    def _llm_hidden(self, batch: Dict[str, Tensor], features: Dict[str, Tensor | list[int]],
                    route_features: Tensor) -> Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        n = input_ids.shape[0]
        additions = {
            "audio_positions": "audio", "visual_positions": "visual", "text_positions": "text",
            "graph_positions": "graph",
        }
        pooled_chunks = []
        for start in range(0, n, self.llm_micro_batch):
            end = min(n, start + self.llm_micro_batch)
            # Keep the LLM call chunked to fit a 12GB GPU. Each chunk retains its
            # autograd graph and therefore still updates LoRA/adapters jointly.
            ids = input_ids[start:end]
            mask = attention_mask[start:end]
            chunk = self.llm.get_input_embeddings()(ids).clone()
            for pos_key, feat_name in additions.items():
                pos = batch[pos_key][start:end]
                feat = self.modality_to_llm[feat_name](features[feat_name][start:end])
                rows = torch.arange(end - start, device=chunk.device)
                chunk[rows, pos] = chunk[rows, pos] + self.feature_scale * feat
            route = self.route_to_llm(route_features[start:end])
            route_pos = batch["route_positions"][start:end]
            rows = torch.arange(end - start, device=chunk.device)
            chunk[rows, route_pos] = chunk[rows, route_pos] + self.feature_scale * route
            outputs = self.llm(inputs_embeds=chunk, attention_mask=mask,
                               output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs.hidden_states[-1]
            last = mask.sum(dim=1).long().clamp_min(1) - 1
            pooled_chunks.append(hidden[torch.arange(end - start, device=hidden.device), last])
        return torch.cat(pooled_chunks, dim=0)

    def forward(self, batch: Dict[str, Tensor], route_features: Tensor | None = None,
                labels: Tensor | None = None) -> Dict[str, Tensor]:
        features = self.encode_modalities(batch)
        n = features["graph"].shape[0]
        if route_features is None:
            route_features = batch.get("route_features")
        if route_features is None:
            route_features = features["graph"].new_tensor([0.5, 0.5, 0, 0, 0.5]).repeat(n, 1)
        route_features = route_features.to(features["graph"].device, dtype=features["graph"].dtype)
        raw_graph = features["graph"]
        local = (features["text"] + features["audio"] + features["visual"]) / 3.0
        alpha = route_features[:, 0:1]
        routed_graph = alpha * features["graph"] + (1.0 - alpha) * local
        features["graph"] = routed_graph
        aux = torch.stack([head(features[name]) for head, name in zip(
            self.aux_heads, ("text", "audio", "visual", "graph"))], dim=1)
        pooled = self._llm_hidden(batch, features, route_features)
        logits = self.classifier(pooled.float())
        result = {"logits": logits, "aux_logits": aux, "route_features": route_features,
                  "features_text": features["text"], "features_audio": features["audio"],
                  "features_visual": features["visual"], "features_graph": features["graph"]}
        if labels is not None:
            ce = F.cross_entropy(logits, labels)
            aux_ce = torch.stack([F.cross_entropy(aux[:, i], labels) for i in range(4)]).mean()
            hgr = soft_hgr_loss([features["text"], features["audio"],
                                 features["visual"], raw_graph])
            result["loss"] = ce + self.lambda_aux * aux_ce + self.lambda_hgr * hgr
            result["loss_ce"] = ce.detach()
            result["loss_aux"] = aux_ce.detach()
            result["loss_hgr"] = hgr.detach()
        return result
