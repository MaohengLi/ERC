"""Check that memory-bounded chunks preserve multimodal and backbone gradients."""

import copy
from types import SimpleNamespace
import torch
from torch import nn
from causal_dialoguegcn_mllm_paper_aligned_4b.model import CausalDialogueGCNLLM


class CausalToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(24, 8)
        self.mix = nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        states = self.mix(inputs_embeds.cumsum(1))
        return SimpleNamespace(last_hidden_state=states)


def test_chunk_checkpoint_matches_direct_gradients():
    torch.manual_seed(19)
    model = CausalDialogueGCNLLM(CausalToyLM(), (1, 2, 3), text_dim=4, audio_dim=4,
                                visual_dim=4, hidden_dim=8, num_heads=2,
                                graph_layers=1, dropout=0, llm_micro_batch=2)
    direct = copy.deepcopy(model).eval()
    model.train()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]] * 3),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1]*6]),
             "audio_positions": torch.zeros(3, dtype=torch.long),
             "visual_positions": torch.ones(3, dtype=torch.long),
             "text_positions": torch.full((3,), 2, dtype=torch.long)}
    features = {k: torch.randn(3, 8, requires_grad=True) for k in ("audio", "visual", "text")}
    other = {k: v.detach().clone().requires_grad_() for k, v in features.items()}
    out = model._llm_hidden(batch, features)
    ref = direct._llm_hidden(batch, other)
    torch.testing.assert_close(out, ref)
    weights = torch.randn_like(out)
    (out * weights).sum().backward()
    (ref * weights).sum().backward()
    for key in features:
        assert features[key].grad.abs().sum() > 0
        torch.testing.assert_close(features[key].grad, other[key].grad)
    for (name, parameter), (other_name, expected) in zip(model.named_parameters(), direct.named_parameters()):
        assert name == other_name
        if expected.grad is not None:
            torch.testing.assert_close(parameter.grad, expected.grad)
