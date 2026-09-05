"""Train Causal-ERC DialogueGCN Qwen3-4B from config.yaml."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,yaml
from data import LABELS,load_iemocap,collate_dialogues
from models import CausalERCDialogueGCNQwen3,load_qwen3_4b
from utils.seed import seed_everything
from utils.metrics import classification_metrics
from utils.checkpoint import save_checkpoint

TOKENS=("<|audio_feat|>","<|visual_feat|>","<|text_feat|>")

def load_config(path):
    with Path(path).open("r",encoding="utf-8") as stream: return yaml.safe_load(stream)

def build_tokenizer(name):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"; tok.truncation_side="left"; tok.add_special_tokens({"additional_special_tokens":list(TOKENS)})
    return tok,tuple(tok.convert_tokens_to_ids(x) for x in TOKENS)

def move(batch,device): return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}

def build_model(cfg,tokenizer,token_ids,sample,training=True):
    lc=cfg["lora"]; mc=cfg["model"]
    llm=load_qwen3_4b(mc["backbone"],len(tokenizer),training,lc["rank"],lc["alpha"],lc["dropout"])
    return CausalERCDialogueGCNQwen3(llm,token_ids,len(sample["text_feats"][0]),len(sample["audio_feats"][0]),len(sample["visual_feats"][0]),hidden_dim=mc["hidden_dim"],num_classes=mc["num_classes"],heads=mc["attention_heads"],dropout=mc["dropout"],lambda_hgr=cfg["loss"]["lambda_hgr"],gcn_layers=mc["gcn_layers"],context_window=mc["graph_context_window"],history_window=mc["prompt_history_window"]).cuda()

@torch.no_grad()
def evaluate_dialogues(model,dialogues,tokenizer,token_ids,cfg):
    model.eval(); y=[]; pred=[]; tc=cfg["training"]; mc=cfg["model"]
    for start in range(0,len(dialogues),tc["dialogue_batch_size"]):
        ds=dialogues[start:start+tc["dialogue_batch_size"]]; batch=move(collate_dialogues(ds,tokenizer,token_ids,tc["max_length"],mc["prompt_history_window"],standard=True),torch.device("cuda")); out=model(batch,batch["labels"],tokenizer=tokenizer,dialogues=ds,max_length=tc["max_length"],history_window=mc["prompt_history_window"]); y.extend(batch["labels"].cpu().tolist()); pred.extend(out["logits"].argmax(-1).cpu().tolist())
    return classification_metrics(y,pred),y,pred

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config.yaml"); args=parser.parse_args(); cfg=load_config(args.config); tc=cfg["training"]; seed_everything(tc["seed"]); data=load_iemocap(cfg["data"]["path"]); tokenizer,token_ids=build_tokenizer(cfg["model"]["backbone"]); model=build_model(cfg,tokenizer,token_ids,data["train"][0],True); print(json.dumps(model.parameter_report(),indent=2)); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tc["learning_rate"],weight_decay=tc["weight_decay"]); output=Path(tc["output_dir"]); output.mkdir(parents=True,exist_ok=True); best=-1.; history=[]
    for epoch in range(1,tc["epochs"]+1):
        model.train(); optimizer.zero_grad(set_to_none=True); total=0.; steps=0
        for start in range(0,len(data["train"]),tc["dialogue_batch_size"]):
            ds=data["train"][start:start+tc["dialogue_batch_size"]]; batch=move(collate_dialogues(ds,tokenizer,token_ids,tc["max_length"],cfg["model"]["prompt_history_window"],standard=True),torch.device("cuda")); out=model(batch,batch["labels"],tokenizer=tokenizer,dialogues=ds,max_length=tc["max_length"],history_window=cfg["model"]["prompt_history_window"]); (out["loss"]/tc["gradient_accumulation"]).backward(); total+=float(out["loss"]); steps+=1
            if steps%tc["gradient_accumulation"]==0 or start+tc["dialogue_batch_size"]>=len(data["train"]): torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics,_,_=evaluate_dialogues(model,data["dev"],tokenizer,token_ids,cfg); record={"epoch":epoch,"train_loss":total/max(steps,1),**{f"dev_{k}":v for k,v in metrics.items()}}; history.append(record); print(json.dumps(record),flush=True)
        if metrics["weighted_f1"]>best: best=metrics["weighted_f1"]; save_checkpoint(model,tokenizer,output/"best_model",{"config":cfg,"best_epoch":epoch,"labels":LABELS})
        (output/"training_log.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

if __name__=="__main__": main()
