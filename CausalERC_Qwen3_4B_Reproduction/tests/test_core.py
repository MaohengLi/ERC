import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.dialogue_rnn import DialogueRNN
from models.soft_hgr import soft_hgr_loss
from utils.prompt import build_prompt
from utils.causal_reasoner import CausalReasoner
from data.dataset import collate_dialogues
from models.causal_erc import CausalERCQwen3
def test_rnn_shape():
    y=DialogueRNN(8,12)(torch.randn(2,4,8),torch.zeros(2,4,dtype=torch.long),torch.tensor([4,2])); assert y.shape==(2,4,12)
def test_hgr_finite(): assert torch.isfinite(soft_hgr_loss(torch.randn(5,12),torch.randn(5,12),torch.randn(5,12)))
def test_prompt_tokens():
    p=build_prompt(["A","B"],["hi","bye"],1,["angry"]); assert all(p.count(x)==1 for x in ("<|audio_feat|>","<|visual_feat|>","<|text_feat|>"))
def test_batch_causal_reasoner():
    d={"vid":"d1","utt_count":2,"speakers":["A","B"],"texts":["hello","why?!"]}
    out=CausalReasoner().decide_batch([d],torch.randn(2,6),torch.randn(2,12)); assert len(out["d1"])==2 and out["d1"][0].causal_type in (0,1)

class _Tokenizer:
    pad_token_id=0; eos_token_id=0
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True): return messages[0]["content"]
    def __call__(self,text,**kwargs): return {"input_ids":torch.tensor([[1,11,12,13,2,3]])}

class _LLM(torch.nn.Module):
    class Config: hidden_size=16
    config=Config()
    def __init__(self): super().__init__(); self.emb=torch.nn.Embedding(32,16)
    def get_input_embeddings(self): return self.emb
    def forward(self,inputs_embeds,**kwargs): return type("Out",(),{"hidden_states":(inputs_embeds,inputs_embeds)})()

def test_model_owns_two_stage_flow():
    d={"vid":"d1","utt_count":3,"text_feats":[torch.randn(4) for _ in range(3)],"audio_feats":[torch.randn(4) for _ in range(3)],"visual_feats":[torch.randn(4) for _ in range(3)],"speakers":["A","B","A"],"labels":[0,1,2],"texts":["hello","why?!","fine"]}
    tok=_Tokenizer(); batch=collate_dialogues([d],tok,(11,12,13),max_length=16,standard=True); model=CausalERCQwen3(_LLM(),(11,12,13),4,4,4,hidden_dim=8,heads=2,dropout=0.0); out=model(batch,batch["labels"],tokenizer=tok,dialogues=[d],max_length=16); assert out["logits"].shape==(3,6) and out["initial_logits"].shape==(3,6) and out["loss"].isfinite()
