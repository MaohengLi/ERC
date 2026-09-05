import argparse,json
from pathlib import Path
import torch
from data import LABELS,load_iemocap,collate_dialogues
from models import CausalERCQwen3,load_qwen3_4b
from utils.seed import seed_everything
from utils.causal_reasoner import CausalReasoner
from utils.checkpoint import save_checkpoint
from utils.metrics import classification_metrics
DEFAULT_DATA=r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN\MultiModal_DialogGCN\save\data.pkl"; TOKENS=("<|audio_feat|>","<|visual_feat|>","<|text_feat|>")
def build_tokenizer(name):
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    if t.pad_token is None:t.pad_token=t.eos_token
    t.padding_side="right"; t.truncation_side="left"; t.add_special_tokens({"additional_special_tokens":list(TOKENS)}); return t,tuple(t.convert_tokens_to_ids(x) for x in TOKENS)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data_path",default=DEFAULT_DATA); p.add_argument("--model_name",default="Qwen/Qwen3-4B"); p.add_argument("--output_dir",default="outputs/qwen3_4b"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch_size",type=int,default=1); p.add_argument("--grad_accumulation",type=int,default=4); p.add_argument("--max_length",type=int,default=768); p.add_argument("--history_window",type=int,default=4); p.add_argument("--lambda_hgr",type=float,default=.1); p.add_argument("--learning_rate",type=float,default=2e-4); p.add_argument("--seed",type=int,default=42); p.add_argument("--max_train_dialogues",type=int,default=0); a=p.parse_args(); seed_everything(a.seed)
    data=load_iemocap(a.data_path); train=data["train"][:a.max_train_dialogues or None]; tok,ids=build_tokenizer(a.model_name); s=train[0]; model=CausalERCQwen3(load_qwen3_4b(a.model_name,len(tok),True),ids,len(s["text_feats"][0]),len(s["audio_feats"][0]),len(s["visual_feats"][0]),lambda_hgr=a.lambda_hgr).cuda(); print(json.dumps(model.parameter_report(),indent=2)); opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.learning_rate); reasoner=CausalReasoner(); best=-1.; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); y=[]; pred=[]; total=0.
        for step in range(0,len(train),a.batch_size):
            ds=train[step:step+a.batch_size]
            initial=collate_dialogues(ds,tok,ids,a.max_length,a.history_window,standard=True); initial={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in initial.items()}
            first=model.initial_forward(initial); lengths=initial["lengths"]; emotion=torch.cat([first["context"]["multimodal"][b,:int(lengths[b])] for b in range(len(ds))]); decisions=reasoner.decide_batch(ds,first["initial_logits"],emotion)
            b=collate_dialogues(ds,tok,ids,a.max_length,a.history_window,decisions=decisions); b={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in b.items()}; o=model(b,b["labels"]); (o["loss"]/a.grad_accumulation).backward(); total+=float(o["loss"]); pred+=o["logits"].argmax(-1).detach().cpu().tolist(); y+=b["labels"].cpu().tolist()
            if ((step//a.batch_size)+1)%a.grad_accumulation==0 or step+a.batch_size>=len(train): opt.step(); opt.zero_grad(set_to_none=True)
        m=classification_metrics(y,pred); rec={"epoch":epoch,"loss":total/max(1,len(train)),**m}; history.append(rec); print(json.dumps(rec),flush=True)
        if m["weighted_f1"]>best: best=m["weighted_f1"]; save_checkpoint(model,tok,Path(a.output_dir)/"best_model",{"args":vars(a),"labels":LABELS})
    Path(a.output_dir).mkdir(parents=True,exist_ok=True); (Path(a.output_dir)/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
if __name__=="__main__": main()
