import argparse,json
from pathlib import Path
import torch
from data import LABELS,load_iemocap,collate_dialogues
from models import CausalERCQwen3,load_qwen3_4b
from train import build_tokenizer,DEFAULT_DATA
from utils.checkpoint import load_non_llm
from utils.metrics import classification_metrics,report
from utils.causal_reasoner import CausalReasoner
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data_path",default=DEFAULT_DATA); p.add_argument("--model_name",default="Qwen/Qwen3-4B"); p.add_argument("--checkpoint_dir",default="outputs/qwen3_4b/best_model"); p.add_argument("--max_length",type=int,default=768); a=p.parse_args(); d=load_iemocap(a.data_path); tok,ids=build_tokenizer(a.model_name); s=d["train"][0]; m=CausalERCQwen3(load_qwen3_4b(a.model_name,len(tok),False),ids,len(s["text_feats"][0]),len(s["audio_feats"][0]),len(s["visual_feats"][0])).cuda(); load_non_llm(m,Path(a.checkpoint_dir)/"non_llm.pt"); m.eval(); y=[]; pred=[]
    reasoner=CausalReasoner()
    with torch.no_grad():
        for x in d["test"]:
            initial=collate_dialogues([x],tok,ids,a.max_length,standard=True); initial={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in initial.items()}; first=m.initial_forward(initial); n=x["utt_count"]; emotion=first["context"]["multimodal"][0,:n]; decisions=reasoner.decide_batch([x],first["initial_logits"],emotion); b=collate_dialogues([x],tok,ids,a.max_length,decisions=decisions); y+=b["labels"].tolist(); b={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in b.items()}; pred+=m(b)["logits"].argmax(-1).cpu().tolist()
    result=classification_metrics(y,pred); result["classification_report"]=report(y,pred,LABELS); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
