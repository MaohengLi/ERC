"""4B-compatible implementation of the Causal-ERC multimodal backbone."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DialogueRNN(nn.Module):
    """Speaker-aware two-layer GRU run independently for every dialogue."""

    def __init__(self, input_dim: int, hidden_dim: int = 200,
                 max_speakers: int = 16, dropout: float = 0.3):
        super().__init__()
        self.speaker_embedding = nn.Embedding(max_speakers, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2,
                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, speaker_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features + self.speaker_embedding(speaker_ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True,
                                        total_length=features.size(1))
        return output


class CrossAttentionFusion(nn.Module):
    """Equations 3-5: each modality attends over the three modality tokens."""

    def __init__(self, hidden_dim: int = 200, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a: torch.Tensor, h_v: torch.Tensor,
                h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = h_a.shape
        modalities = torch.stack([h_a, h_v, h_t], dim=2).reshape(-1, 3, shape[-1])
        a, _ = self.attn_a(modalities[:, :1], modalities, modalities)
        v, _ = self.attn_v(modalities[:, 1:2], modalities, modalities)
        t, _ = self.attn_t(modalities[:, 2:3], modalities, modalities)
        f_a = self.norm_a(modalities[:, 0] + self.dropout(a[:, 0])).view(shape)
        f_v = self.norm_v(modalities[:, 1] + self.dropout(v[:, 0])).view(shape)
        f_t = self.norm_t(modalities[:, 2] + self.dropout(t[:, 0])).view(shape)
        return f_a, f_v, f_t


def soft_hgr_loss(*features: torch.Tensor) -> torch.Tensor:
    """Numerically stable, dimension-normalised form of paper Equation 6."""

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
            penalty = 0.5 * (cov_q * cov_v).sum() / dim
            total = total - (correlation - penalty)
            pairs += 1
    return total / pairs


class CausalERC4B(nn.Module):
    """DialogueRNN + cross-attention + multimodal token injection + classifier."""

    def __init__(self, llm: nn.Module, special_token_ids: tuple[int, int, int],
                 text_dim: int = 100, audio_dim: int = 100, video_dim: int = 512,
                 hidden_dim: int = 200, num_classes: int = 6,
                 num_heads: int = 4, dropout: float = 0.3,
                 lambda_hgr: float = 1.0, llm_micro_batch: int = 2):
        super().__init__()
        self.llm = llm
        self.audio_token_id, self.visual_token_id, self.text_token_id = special_token_ids
        self.lambda_hgr = lambda_hgr
        self.llm_micro_batch = llm_micro_batch

        self.rnn_t = DialogueRNN(text_dim, hidden_dim, dropout=dropout)
        self.rnn_a = DialogueRNN(audio_dim, hidden_dim, dropout=dropout)
        self.rnn_v = DialogueRNN(video_dim, hidden_dim, dropout=dropout)
        self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)

        llm_hidden = int(llm.config.hidden_size)
        self.proj_a = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.proj_v = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.proj_t = nn.Sequential(nn.Linear(hidden_dim, llm_hidden), nn.LayerNorm(llm_hidden))
        self.final_norm = nn.LayerNorm(llm_hidden)
        self.classifier = nn.Linear(llm_hidden, num_classes)

    @staticmethod
    def _valid_flatten(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[b, :int(lengths[b].item())] for b in range(x.size(0))], dim=0)

    def encode_modalities(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = batch["lengths"]
        h_t = self.rnn_t(batch["text"], batch["speaker_ids"], lengths)
        h_a = self.rnn_a(batch["audio"], batch["speaker_ids"], lengths)
        h_v = self.rnn_v(batch["visual"], batch["speaker_ids"], lengths)
        f_a, f_v, f_t = self.fusion(h_a, h_v, h_t)
        return tuple(self._valid_flatten(x, lengths) for x in (f_a, f_v, f_t))

    def _llm_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    f_a: torch.Tensor, f_v: torch.Tensor, f_t: torch.Tensor,
                    a_pos: torch.Tensor, v_pos: torch.Tensor,
                    t_pos: torch.Tensor) -> torch.Tensor:
        rows = []
        micro = max(1, self.llm_micro_batch)
        for start in range(0, input_ids.size(0), micro):
            end = min(input_ids.size(0), start + micro)
            ids = input_ids[start:end]
            mask = attention_mask[start:end]
            embeds = self.llm.get_input_embeddings()(ids).clone()
            local = torch.arange(end - start, device=ids.device)
            embeds[local, a_pos[start:end]] = self.proj_a(f_a[start:end]).to(embeds.dtype)
            embeds[local, v_pos[start:end]] = self.proj_v(f_v[start:end]).to(embeds.dtype)
            embeds[local, t_pos[start:end]] = self.proj_t(f_t[start:end]).to(embeds.dtype)
            output = self.llm(
                inputs_embeds=embeds, attention_mask=mask,
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
            last_hidden = output.hidden_states[-1]
            final_pos = mask.sum(-1).long() - 1
            rows.append(last_hidden[local, final_pos].float())
        return torch.cat(rows, dim=0)

    def forward(self, batch: dict, labels: torch.Tensor | None = None) -> dict:
        f_a, f_v, f_t = self.encode_modalities(batch)
        hidden = self._llm_hidden(
            batch["input_ids"], batch["attention_mask"], f_a, f_v, f_t,
            batch["audio_positions"], batch["visual_positions"], batch["text_positions"],
        )
        logits = self.classifier(self.final_norm(hidden))
        loss_hgr = soft_hgr_loss(f_t, f_a, f_v)
        loss_ce = None if labels is None else F.cross_entropy(logits, labels)
        loss = None if loss_ce is None else loss_ce + self.lambda_hgr * loss_hgr
        return {
            "logits": logits, "loss": loss, "loss_ce": loss_ce,
            "loss_hgr": loss_hgr, "f_t": f_t, "f_a": f_a, "f_v": f_v,
        }

    def non_llm_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if not k.startswith("llm.")}

    def save_non_llm(self, path: str | Path) -> None:
        torch.save(self.non_llm_state_dict(), Path(path))

    def load_non_llm(self, path: str | Path) -> None:
        missing, unexpected = self.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False), strict=False)
        missing = [name for name in missing if not name.startswith("llm.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def parameter_report(self) -> dict[str, int]:
        groups = {
            "llm": self.llm,
            "dialogue_rnn": nn.ModuleList([self.rnn_t, self.rnn_a, self.rnn_v]),
            "cross_attention": self.fusion,
            "projectors": nn.ModuleList([self.proj_a, self.proj_v, self.proj_t]),
            "classifier": nn.ModuleList([self.final_norm, self.classifier]),
        }
        report = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        report["total"] = sum(p.numel() for p in self.parameters())
        return report
