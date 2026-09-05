import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.graph_builder import DialogueGraphBuilder,SAME_SPEAKER,INTER_SPEAKER,TEMPORAL
from models.dialogue_gcn import DialogueGCN
from models.multimodal_encoder import MultimodalEncoder
from models.causal_erc import CausalERCDialogueGCNQwen3
from data.dataset import collate_dialogues

def test_graph_relations_are_causal():
    graph=DialogueGraphBuilder(4)(torch.tensor([[0,1,0]]),torch.tensor([3])); src,dst=graph.edge_index
    assert torch.all(src<dst); assert set(graph.edge_type.tolist())=={SAME_SPEAKER,INTER_SPEAKER,TEMPORAL}

def test_dialogue_gcn_shape_and_gradients():
    graph=DialogueGraphBuilder(4)(torch.tensor([[0,1,0]]),torch.tensor([3])); x=torch.randn(3,8,requires_grad=True); y=DialogueGCN(8,2,0.0)(x,graph); y.sum().backward(); assert y.shape==(3,8) and x.grad is not None

def test_three_modality_gcns_are_independent():
    model=MultimodalEncoder(4,4,4,8,2,0.0); ids={id(p) for p in model.gcn_t.parameters()}; assert ids.isdisjoint({id(p) for p in model.gcn_a.parameters()}) and ids.isdisjoint({id(p) for p in model.gcn_v.parameters()})

class Tokenizer:
    pad_token_id=0; eos_token_id=0
    def apply_chat_template(self,messages,**kwargs): return messages[0]["content"]
    def __call__(self,text,**kwargs): return {"input_ids":torch.tensor([[1,11,12,13,2]])}
class LLM(torch.nn.Module):
    class Config: hidden_size=16
    config=Config()
    def __init__(self): super().__init__(); self.emb=torch.nn.Embedding(32,16)
    def get_input_embeddings(self): return self.emb
    def forward(self,inputs_embeds,**kwargs): return type("Out",(),{"hidden_states":(inputs_embeds,)})()

def test_complete_two_stage_model_flow():
    d={"vid":"d","utt_count":3,"text_feats":[torch.randn(4) for _ in range(3)],"audio_feats":[torch.randn(4) for _ in range(3)],"visual_feats":[torch.randn(4) for _ in range(3)],"speakers":["A","B","A"],"labels":[0,1,2],"texts":["a","b","c"]}; tok=Tokenizer(); batch=collate_dialogues([d],tok,(11,12,13),16,standard=True); model=CausalERCDialogueGCNQwen3(LLM(),(11,12,13),4,4,4,8,heads=2,dropout=0.0); out=model(batch,batch["labels"],tok,[d],16,4); assert out["logits"].shape==(3,6) and out["initial_logits"].shape==(3,6) and torch.isfinite(out["loss"])
