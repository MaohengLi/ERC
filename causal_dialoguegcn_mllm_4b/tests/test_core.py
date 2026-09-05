from types import SimpleNamespace

import torch
from torch import nn

from causal_dialoguegcn_mllm_4b.causal_router import (
    DualSystemDecision, classify_multimodal_dialogue, decisions_to_features,
)
from causal_dialoguegcn_mllm_4b.data_dual import SPECIAL_TOKENS, build_dual_prompt
from causal_dialoguegcn_mllm_4b.model import CausalDialogueGCNLLM, DialogueGCNEncoder


def _dialogue(n=3):
    return {
        "vid": "toy", "utt_count": n,
        "speakers": ["A", "B", "A"][:n],
        "texts": ["hello", "not good", "fine now"][:n],
        "labels": [2, 5, 2][:n],
    }


def test_prompt_has_exactly_five_injection_tokens():
    decision = DualSystemDecision(1, 0.2, 0.8, 0.1, 0.7, 0.3, 0, 0, 0.9)
    prompt = build_dual_prompt(_dialogue(), 2, decision)
    assert all(prompt.count(token) == 1 for token in SPECIAL_TOKENS)
    assert "System 1" in prompt and "Peak evidence" in prompt


def test_router_is_future_leak_free_and_bounded():
    torch.manual_seed(3)
    logits = torch.randn(7, 4, 6)
    changed = logits.clone()
    changed[5:] = 100 * torch.randn_like(changed[5:])
    first = classify_multimodal_dialogue(logits)
    second = classify_multimodal_dialogue(changed)
    for t in range(5):
        assert first[t] == second[t]
    route = decisions_to_features(first, 7)
    assert route.shape == (7, 5)
    assert torch.all((route[:, 0] >= 0) & (route[:, 0] <= 1))


def test_c1_c2_distance_semantics_match_causal_erc():
    # Same history mean as target -> C1.  Peak/end prototype is deliberately
    # different, so the paper's smaller-distance rule is unambiguous.
    p_peak = torch.tensor([.10, .10, .70, .03, .04, .03])
    p_end = torch.tensor([.05, .05, .05, .80, .03, .02])
    p_mean = (p_peak + p_end) / 2
    probs = torch.stack((p_peak, p_end, p_mean))
    logits = probs.clamp_min(1e-6).log().unsqueeze(1).repeat(1, 4, 1)
    decisions = classify_multimodal_dialogue(logits, base_history_window=1,
                                              max_history_window=2)
    assert decisions[2].lambda_c1 < decisions[2].lambda_c2
    assert decisions[2].causal_type == 0  # C1, not C2


def test_dialogue_graph_does_not_mix_conversations():
    torch.manual_seed(5)
    encoder = DialogueGCNEncoder(4, 3, 5, hidden_dim=8, num_heads=2,
                                 layers=2, graph_window=2, dropout=0).eval()
    text, audio, visual = torch.randn(2, 3, 4), torch.randn(2, 3, 3), torch.randn(2, 3, 5)
    speakers = torch.tensor([[0, 1, 0], [0, 1, 0]])
    lengths = torch.tensor([3, 3])
    before = encoder(text, audio, visual, speakers, lengths)["graph"][:3]
    text[1] += 100; audio[1] -= 100; visual[1] *= 50
    after = encoder(text, audio, visual, speakers, lengths)["graph"][:3]
    assert torch.allclose(before, after, atol=1e-6)


def test_causal_graph_does_not_read_future_utterances():
    torch.manual_seed(6)
    encoder = DialogueGCNEncoder(4, 3, 5, hidden_dim=8, num_heads=2,
                                 layers=2, graph_window=3, dropout=0,
                                 causal_graph=True).eval()
    text, audio, visual = torch.randn(1, 3, 4), torch.randn(1, 3, 3), torch.randn(1, 3, 5)
    speakers, lengths = torch.tensor([[0, 1, 0]]), torch.tensor([3])
    before = encoder(text, audio, visual, speakers, lengths)["graph"][0]
    text[:, 1:] += 100; audio[:, 1:] -= 100; visual[:, 1:] *= 50
    after = encoder(text, audio, visual, speakers, lengths)["graph"][0]
    assert torch.allclose(before, after, atol=1e-6)


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


def test_complete_forward_loss_and_gradients():
    torch.manual_seed(7)
    model = CausalDialogueGCNLLM(
        DummyLLM(), {"audio": 1, "visual": 2, "text": 3, "graph": 4, "route": 5},
        text_dim=4, audio_dim=3, visual_dim=5, hidden_dim=8, num_heads=2,
        graph_layers=1, graph_window=2, dropout=0, llm_micro_batch=2,
    )
    batch = {
        "text": torch.randn(1, 3, 4), "audio": torch.randn(1, 3, 3),
        "visual": torch.randn(1, 3, 5), "speaker_ids": torch.tensor([[0, 1, 0]]),
        "lengths": torch.tensor([3]), "input_ids": torch.randint(6, 40, (3, 9)),
        "attention_mask": torch.ones(3, 9, dtype=torch.long),
        "audio_positions": torch.tensor([0, 0, 0]),
        "visual_positions": torch.tensor([1, 1, 1]),
        "text_positions": torch.tensor([2, 2, 2]),
        "graph_positions": torch.tensor([3, 3, 3]),
        "route_positions": torch.tensor([4, 4, 4]),
        "labels": torch.tensor([0, 2, 5]),
        "route_features": torch.rand(3, 5),
    }
    output = model(batch, labels=batch["labels"])
    assert output["logits"].shape == (3, 6)
    assert output["aux_logits"].shape == (3, 4, 6)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.encoder.node_fuse[0].weight.grad is not None
    assert model.route_to_llm[1].weight.grad is not None
    assert model.llm.block.weight.grad is not None
