from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from causal_erc_repro_4b.causal_prompting import build_erc_prompt, classify_dialogue
from causal_erc_repro_4b.model import CausalERC4B, DialogueRNN, soft_hgr_loss


class DummyLLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.block = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        hidden = self.block(inputs_embeds)
        return SimpleNamespace(hidden_states=(hidden,))


class CoreTests(unittest.TestCase):
    def test_peak_end_classification_and_first_turn(self):
        logits = torch.tensor([
            [1.0, 0.0], [1.1, 0.0], [1.05, 0.0], [3.0, 0.0],
        ])
        decisions = classify_dialogue(logits, base_history_window=2, max_history_window=3)
        self.assertIsNone(decisions[0].causal_type)
        self.assertEqual(decisions[2].causal_type, 0)  # close to history mean
        self.assertEqual(decisions[3].causal_type, 1)  # close to peak/end state
        self.assertLess(decisions[2].lambda_c1, decisions[2].lambda_c2)

    def test_c2_prompt_marks_peak_and_all_modal_tokens(self):
        logits = torch.tensor([[1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        decision = classify_dialogue(logits, 1, 2)[2]
        prompt = build_erc_prompt(
            ["M", "F", "M"], ["first", "peak", "target"], 2,
            ["happy", "sad"], decision,
        )
        self.assertIn("Peak utterance", prompt)
        self.assertIn("<|audio_feat|>", prompt)
        self.assertIn("<|visual_feat|>", prompt)
        self.assertIn("<|text_feat|>", prompt)

    def test_dialogue_rnn_preserves_batch_boundaries(self):
        rnn = DialogueRNN(4, hidden_dim=6, max_speakers=2, dropout=0.0)
        x = torch.randn(2, 3, 4)
        speaker = torch.zeros(2, 3, dtype=torch.long)
        out = rnn(x, speaker, torch.tensor([3, 1]))
        self.assertEqual(tuple(out.shape), (2, 3, 6))
        self.assertTrue(torch.equal(out[1, 1:], torch.zeros_like(out[1, 1:])))

    def test_full_model_shapes_and_loss(self):
        model = CausalERC4B(
            DummyLLM(), (10, 11, 12), text_dim=4, audio_dim=3,
            video_dim=5, hidden_dim=8, num_classes=6, num_heads=2,
            dropout=0.0, llm_micro_batch=1,
        )
        ids = torch.tensor([[1, 10, 11, 12, 2], [1, 10, 11, 12, 2]])
        batch = {
            "text": torch.randn(1, 2, 4),
            "audio": torch.randn(1, 2, 3),
            "visual": torch.randn(1, 2, 5),
            "speaker_ids": torch.tensor([[0, 1]]),
            "lengths": torch.tensor([2]),
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "audio_positions": torch.tensor([1, 1]),
            "visual_positions": torch.tensor([2, 2]),
            "text_positions": torch.tensor([3, 3]),
        }
        result = model(batch, labels=torch.tensor([0, 1]))
        self.assertEqual(tuple(result["logits"].shape), (2, 6))
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_soft_hgr_single_item_is_zero(self):
        x = torch.randn(1, 8)
        self.assertEqual(float(soft_hgr_loss(x, x, x)), 0.0)


if __name__ == "__main__":
    unittest.main()
