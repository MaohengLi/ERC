import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.dialogue_rnn import DialogueRNN
from models.soft_hgr import soft_hgr_loss
from utils.prompt import build_prompt,select_causal_type
from utils.causal_reasoner import CausalReasoner
def test_rnn_shape():
    y=DialogueRNN(8,12)(torch.randn(2,4,8),torch.zeros(2,4,dtype=torch.long),torch.tensor([4,2])); assert y.shape==(2,4,12)
def test_hgr_finite(): assert torch.isfinite(soft_hgr_loss(torch.randn(5,12),torch.randn(5,12),torch.randn(5,12)))
def test_prompt_tokens():
    p=build_prompt(["A","B"],["hi","bye"],1,["angry"]); assert all(p.count(x)==1 for x in ("<|audio_feat|>","<|visual_feat|>","<|text_feat|>"))
def test_causal_context_selection(): assert select_causal_type(["A"],["hello"],0).causal_type in (0,1)
def test_batch_causal_reasoner():
    d={"vid":"d1","utt_count":2,"speakers":["A","B"],"texts":["hello","why?!"]}
    out=CausalReasoner().decide_batch([d],torch.randn(2,6),torch.randn(2,12)); assert len(out["d1"])==2 and out["d1"][0].causal_type in (0,1)
