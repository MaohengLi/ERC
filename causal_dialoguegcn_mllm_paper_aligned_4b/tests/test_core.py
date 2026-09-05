from types import SimpleNamespace

import torch
from torch import nn

from causal_dialoguegcn_mllm_paper_aligned_4b.causal_router import (
    DualSystemDecision, classify_multimodal_dialogue,
)
from causal_dialoguegcn_mllm_paper_aligned_4b.data import SPECIAL_TOKENS, build_causal_prompt
from causal_dialoguegcn_mllm_paper_aligned_4b.model import CausalDialogueGCNLLM, DialogueGCNEncoder


def toy_dialogue(n=3):
    return {"vid": "toy", "utt_count": n, "speakers": ["A", "B", "A"][:n],
            "texts": ["hello", "not good", "fine now"][:n], "labels": [2, 5, 2][:n]}


def test_original_interface_has_only_three_feature_tokens():
    decision = DualSystemDecision(causal_type=1, lambda_c1=0.2,
                                  lambda_c2=0.8, peak_idx=0,
                                  history_start=0, alpha_system2=0.2,
                                  reliability=0.8, disagreement=0,
                                  peak_intensity=0.9)
    prompt = build_causal_prompt(toy_dialogue(), 2, decision)
    assert all(prompt.count(token) == 1 for token in SPECIAL_TOKENS)
    assert "<|graph_feat|>" not in prompt
    assert "<|causal_route|>" not in prompt
    assert "Causal type C2" in prompt


def test_c1_c2_rule_uses_paper_smaller_distance_semantics():
    p_first = torch.tensor([.10, .10, .70, .03, .04, .03])
    p_second = torch.tensor([.05, .05, .05, .80, .03, .02])
    target = (p_first + p_second) / 2
    logits = torch.stack((p_first, p_second, target)).clamp_min(1e-6).log()
    logits = logits.unsqueeze(1).repeat(1, 4, 1)
    decisions = classify_multimodal_dialogue(logits, 1, 2)
    assert decisions[2].lambda_c1 < decisions[2].lambda_c2
    assert decisions[2].causal_type == 0


def test_causal_graph_is_dialogue_local_and_past_only():
    torch.manual_seed(4)
    encoder = DialogueGCNEncoder(4, 3, 5, hidden_dim=8, num_heads=2,
                                 layers=1, graph_window=2, dropout=0,
                                 causal_graph=True).eval()
    text, audio, visual = torch.randn(2, 3, 4), torch.randn(2, 3, 3), torch.randn(2, 3, 5)
    speaker_ids, lengths = torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([3, 3])
    before = encoder(text, audio, visual, speaker_ids, lengths)["graph"][:3]
    text[0, 1:] += 100; audio[0, 1:] -= 100; visual[0, 1:] *= 50
    after = encoder(text, audio, visual, speaker_ids, lengths)["graph"][:3]
    assert torch.allclose(before[:1], after[:1], atol=1e-6)
    before_second = encoder(text, audio, visual, speaker_ids, lengths)["graph"][3:]
    text[1] += 100
    after_second = encoder(text, audio, visual, speaker_ids, lengths)["graph"][3:]
    assert not torch.allclose(before_second, after_second)


class DummyLLM(nn.Module):
    def __init__(self, vocab=40, hidden=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.block = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.block(inputs_embeds))


def test_forward_has_three_tokens_and_gate_gradients():
    torch.manual_seed(8)
    model = CausalDialogueGCNLLM(
        DummyLLM(), (1, 2, 3), text_dim=4, audio_dim=3, visual_dim=5,
        hidden_dim=8, num_heads=2, graph_layers=1, graph_window=2,
        dropout=0, llm_micro_batch=2,
    )
    batch = {
        "text": torch.randn(1, 3, 4), "audio": torch.randn(1, 3, 3),
        "visual": torch.randn(1, 3, 5), "speaker_ids": torch.tensor([[0, 1, 0]]),
        "lengths": torch.tensor([3]), "input_ids": torch.randint(4, 40, (3, 8)),
        "attention_mask": torch.ones(3, 8, dtype=torch.long),
        "audio_positions": torch.tensor([0, 0, 0]),
        "visual_positions": torch.tensor([1, 1, 1]),
        "text_positions": torch.tensor([2, 2, 2]),
        "labels": torch.tensor([0, 2, 5]), "route_features": torch.rand(3, 5),
    }
    output = model(batch, labels=batch["labels"])
    assert output["logits"].shape == (3, 6)
    assert output["route_logits"].shape == (3, 4, 6)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.encoder.node_fuse[0].weight.grad is not None
    assert model.graph_to_modality["text"][1].weight.grad is not None
    assert model.llm.block.weight.grad is not None
