"""DialogueGCN replacement with the original Causal-ERC LLM interface.

The LLM receives exactly a causal text prompt plus three projected feature
tokens (audio, visual and text).  Graph context and dual-system routing are
used before those three projections; no Graph Token or Route Token is added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def soft_hgr_loss(features: Sequence[Tensor], eps: float = 1e-6) -> Tensor:
    """Stable sample version of the paper's Soft-HGR consistency objective."""
    if len(features) < 2 or features[0].size(0) < 2:
        return features[0].new_zeros(())
    dim = features[0].size(-1)
    total = features[0].new_zeros((), dtype=torch.float32)
    pairs = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            q = features[i].float() - features[i].float().mean(0, keepdim=True)
            v = features[j].float() - features[j].float().mean(0, keepdim=True)
            correlation = (q * v).sum(-1).mean() / dim
            cov_q = q.T @ q / max(q.size(0) - 1, 1)
            cov_v = v.T @ v / max(v.size(0) - 1, 1)
            covariance_penalty = 0.5 * (cov_q * cov_v).sum() / (dim + eps)
            total = total - (correlation - covariance_penalty)
            pairs += 1
    return total / pairs


class DialogueGCNLayer(nn.Module):
    """Speaker-aware local graph attention with optional causal direction."""

    def __init__(self, hidden_dim: int, window: int = 10, dropout: float = 0.2,
                 causal_only: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window = window
        self.causal_only = causal_only
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_weight = nn.Parameter(torch.empty(8, hidden_dim, hidden_dim))
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
    def relation(src: int, dst: int, future: bool) -> int:
        return ((int(src) % 2) * 2 + (int(dst) % 2)) * 2 + int(future)

    def forward(self, x: Tensor, speaker_ids: Tensor) -> Tensor:
        if x.size(0) == 0:
            return x
        q, k = self.q(x), self.k(x)
        speakers = speaker_ids.detach().cpu().tolist()
        messages = []
        scale = self.hidden_dim ** -0.5
        for i in range(x.size(0)):
            lo = max(0, i - self.window)
            hi = i + 1 if self.causal_only else min(x.size(0), i + self.window + 1)
            neighbours = list(range(lo, hi))
            scores = torch.stack([torch.dot(q[i], k[j]) * scale for j in neighbours])
            weights = torch.softmax(scores, dim=0)
            aggregate = x.new_zeros(self.hidden_dim)
            for weight, j in zip(weights, neighbours):
                rel = self.relation(speakers[j], speakers[i], j > i)
                aggregate = aggregate + weight * torch.matmul(self.relation_weight[rel], x[j])
            messages.append(aggregate)
        message = torch.stack(messages)
        y = self.out_norm(x + self.dropout(message))
        return self.ffn_norm(y + self.ffn(y))


class DialogueGCNEncoder(nn.Module):
    """Three modality projections, fusion attention and per-dialogue graph."""

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
            DialogueGCNLayer(hidden_dim, graph_window, dropout, causal_graph)
            for _ in range(layers)
        ])

    def forward(self, text: Tensor, audio: Tensor, visual: Tensor,
                speaker_ids: Tensor, lengths: Tensor) -> Dict[str, Tensor | list[int]]:
        ft, fa, fv = self.text_proj(text), self.audio_proj(audio), self.visual_proj(visual)
        batch, max_len, _ = ft.shape
        modal = torch.stack((ft, fa, fv), dim=2).reshape(batch * max_len, 3, self.hidden_dim)
        attended, _ = self.cross_attn(modal, modal, modal, need_weights=False)
        attended = self.cross_norm(modal + attended).reshape(batch, max_len, 3, self.hidden_dim)
        ft, fa, fv = attended.unbind(dim=2)

        flat_t, flat_a, flat_v, flat_g, counts = [], [], [], [], []
        for b in range(batch):
            n = int(lengths[b].item())
            counts.append(n)
            t_b, a_b, v_b = ft[b, :n], fa[b, :n], fv[b, :n]
            graph = self.node_fuse(torch.cat((t_b, a_b, v_b), dim=-1))
            for layer in self.graph_layers:
                graph = layer(graph, speaker_ids[b, :n])
            flat_t.append(t_b); flat_a.append(a_b); flat_v.append(v_b); flat_g.append(graph)
        return {
            "text": torch.cat(flat_t), "audio": torch.cat(flat_a),
            "visual": torch.cat(flat_v), "graph": torch.cat(flat_g), "counts": counts,
        }


class CausalDialogueGCNLLM(nn.Module):
    """Paper-interface-aligned Causal-ERC extension."""

    def __init__(self, llm: nn.Module, special_token_ids: tuple[int, int, int],
                 text_dim: int = 100, audio_dim: int = 100, visual_dim: int = 512,
                 num_classes: int = 6, hidden_dim: int = 200, num_heads: int = 4,
                 graph_layers: int = 2, graph_window: int = 10, dropout: float = 0.2,
                 lambda_hgr: float = 1.0, lambda_aux: float = 0.1,
                 llm_micro_batch: int = 4, causal_graph: bool = True) -> None:
        super().__init__()
        self.llm = llm
        self.audio_token_id, self.visual_token_id, self.text_token_id = special_token_ids
        self.lambda_hgr, self.lambda_aux = lambda_hgr, lambda_aux
        self.llm_micro_batch = max(1, int(llm_micro_batch))
        self.encoder = DialogueGCNEncoder(
            text_dim, audio_dim, visual_dim, hidden_dim, num_heads,
            graph_layers, graph_window, dropout, causal_graph,
        )
        # These probes are internal supervision for multimodal routing. Their
        # outputs are cached for the next pass but are never LLM input tokens.
        self.aux_heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(3)])
        self.graph_to_modality = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
            for name in ("audio", "visual", "text")
        })
        llm_hidden = int(getattr(llm.config, "hidden_size", 0))
        if llm_hidden <= 0 and hasattr(llm.config, "text_config"):
            llm_hidden = int(llm.config.text_config.hidden_size)
        if llm_hidden <= 0:
            raise ValueError("Unable to infer LLM hidden_size")
        self.llm_hidden = llm_hidden
        self.proj_a = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.proj_v = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.proj_t = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.final_norm = nn.LayerNorm(llm_hidden)
        self.classifier = nn.Linear(llm_hidden, num_classes)

    def parameter_report(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        logical_llm = self.llm.get_nb_trainable_parameters()[1] if hasattr(self.llm, "get_nb_trainable_parameters") else sum(p.numel() for p in self.llm.parameters())
        groups = {"encoder": self.encoder, "graph_to_modality": self.graph_to_modality,
                  "auxiliary_heads": self.aux_heads,
                  "avt_projectors": nn.ModuleList([self.proj_a, self.proj_v, self.proj_t]),
                  "classifier": nn.ModuleList([self.final_norm, self.classifier])}
        return {**{name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()},
                "logical_llm_parameters": logical_llm,
                "packed_storage_numel": total, "trainable": trainable, "llm_trainable": llm_trainable,
                "adapter_trainable": trainable - llm_trainable}

    def save_non_llm(self, path: str | Path) -> None:
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()
                    if not k.startswith("llm.")}, Path(path))

    def load_non_llm(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("llm.")]
        unexpected = [k for k in unexpected if not k.startswith("llm.")]
        if missing or unexpected:
            raise RuntimeError(f"Invalid non-LLM checkpoint: missing={missing}, unexpected={unexpected}")

    def encode_modalities(self, batch: dict) -> Dict[str, Tensor | list[int]]:
        return self.encoder(batch["text"], batch["audio"], batch["visual"],
                             batch["speaker_ids"], batch["lengths"])

    def _llm_hidden(self, batch: dict, features: Dict[str, Tensor | list[int]]) -> Tensor:
        ids, mask = batch["input_ids"], batch["attention_mask"]
        positions = (("audio_positions", "audio", self.proj_a),
                     ("visual_positions", "visual", self.proj_v),
                     ("text_positions", "text", self.proj_t))
        rows = []
        for start in range(0, ids.size(0), self.llm_micro_batch):
            end = min(ids.size(0), start + self.llm_micro_batch)
            local_ids, local_mask = ids[start:end], mask[start:end]
            # Prompts are right-padded. Drop only the padding suffix of this
            # chunk; all actual tokens and their injection positions are kept.
            valid_width = int(local_mask.sum(-1).max().item())
            local_ids, local_mask = local_ids[:, :valid_width], local_mask[:, :valid_width]
            embeds = self.llm.get_input_embeddings()(local_ids).clone()
            local_rows = torch.arange(end - start, device=ids.device)
            for pos_key, name, projector in positions:
                pos = batch[pos_key][start:end]
                value = projector(features[name][start:end]).to(embeds.dtype)
                # Replacement mirrors the original Causal-ERC multimodal token
                # interface rather than introducing additive route embeddings.
                embeds[local_rows, pos] = value
            # Checkpoint the entire chunk, not only individual decoder layers.
            # Otherwise each chunk retains all layer activations until the
            # dialogue-level backward call, which exhausts a 12GB GPU.
            if self.training and torch.is_grad_enabled():
                pooled = checkpoint(self._pooled_decoder, embeds, local_mask,
                                    use_reentrant=True, preserve_rng_state=True)
            else:
                pooled = self._pooled_decoder(embeds, local_mask)
            rows.append(pooled)
        return torch.cat(rows, dim=0)

    def _pooled_decoder(self, embeds: Tensor, mask: Tensor) -> Tensor:
        # Qwen's vocabulary LM head is unused by our six-class classifier.
        # Run the same LoRA-wrapped decoder directly without materialising the
        # [batch, tokens, vocabulary] logits or every layer's hidden state.
        causal_lm = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        decoder = getattr(causal_lm, "model", None)
        if decoder is not None:
            output = decoder(inputs_embeds=embeds, attention_mask=mask,
                             output_hidden_states=False, return_dict=True, use_cache=False)
            hidden = output.last_hidden_state
        else:  # lightweight test backbones
            output = self.llm(inputs_embeds=embeds, attention_mask=mask,
                              output_hidden_states=True, return_dict=True, use_cache=False)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output.hidden_states[-1]
        positions = mask.sum(-1).long().clamp_min(1) - 1
        return hidden[torch.arange(mask.size(0), device=mask.device), positions].float()

    def forward(self, batch: dict, route_features: Tensor | None = None,
                labels: Tensor | None = None) -> Dict[str, Tensor]:
        features = self.encode_modalities(batch)
        n = features["graph"].size(0)
        if route_features is None:
            route_features = batch.get("route_features")
        if route_features is None:
            route_features = features["graph"].new_tensor([0.5, 0.5, 0, 0, 0.5]).repeat(n, 1)
        route_features = route_features.to(features["graph"].device, dtype=features["graph"].dtype)
        alpha = route_features[:, 0:1]
        routed = {}
        for name in ("text", "audio", "visual"):
            graph_context = self.graph_to_modality[name](features["graph"])
            # C1/System 2 (large alpha) favours graph/history context; C2/System
            # 1 (small alpha) favours the local fused utterance representation.
            routed[name] = alpha * graph_context + (1.0 - alpha) * features[name]

        aux = torch.stack([head(routed[name]) for head, name in zip(
            self.aux_heads, ("text", "audio", "visual"))], dim=1)
        hidden = self._llm_hidden(batch, routed)
        logits = self.classifier(self.final_norm(hidden))
        route_logits = torch.cat((aux, logits.unsqueeze(1)), dim=1)
        result = {"logits": logits, "aux_logits": aux, "route_logits": route_logits,
                  "route_features": route_features, "f_text": routed["text"],
                  "f_audio": routed["audio"], "f_visual": routed["visual"]}
        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)
            aux_ce = torch.stack([F.cross_entropy(aux[:, i], labels) for i in range(3)]).mean()
            hgr = soft_hgr_loss([features["text"], features["audio"], features["visual"]])
            result.update({"loss": loss_ce + self.lambda_aux * aux_ce + self.lambda_hgr * hgr,
                           "loss_ce": loss_ce.detach(), "loss_aux": aux_ce.detach(),
                           "loss_hgr": hgr.detach()})
        return result


__all__ = ["soft_hgr_loss", "DialogueGCNLayer", "DialogueGCNEncoder", "CausalDialogueGCNLLM"]
