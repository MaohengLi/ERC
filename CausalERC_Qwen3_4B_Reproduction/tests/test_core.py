import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.dialogue_rnn import DialogueRNN
from models.soft_hgr import soft_hgr_loss
from utils.prompt import build_prompt,classify_dialogue
def test_rnn_shape():
    y=DialogueRNN(8,12)(torch.randn(2,4,8),torch.zeros(2,4,dtype=torch.long),torch.tensor([4,2])); assert y.shape==(2,4,12)
def test_hgr_finite(): assert torch.isfinite(soft_hgr_loss(torch.randn(5,12),torch.randn(5,12),torch.randn(5,12)))
def test_prompt_tokens():
    p=build_prompt(["A","B"],["hi","bye"],1,["angry"]); assert all(p.count(x)==1 for x in ("<|audio_feat|>","<|visual_feat|>","<|text_feat|>"))
def test_causal_first_standard(): assert classify_dialogue(torch.randn(3,6))[0].causal_type is None
