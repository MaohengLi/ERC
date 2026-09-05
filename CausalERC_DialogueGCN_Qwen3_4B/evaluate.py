"""Evaluate the best DialogueGCN baseline checkpoint."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from peft import PeftModel
from data import LABELS,load_iemocap
from models import CausalERCDialogueGCNQwen3,load_qwen3_4b
from train import load_config,build_tokenizer,evaluate_dialogues
from utils.checkpoint import load_non_llm
from utils.metrics import report

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config.yaml"); parser.add_argument("--checkpoint_dir",default=None); args=parser.parse_args(); cfg=load_config(args.config); data=load_iemocap(cfg["data"]["path"]); tokenizer,token_ids=build_tokenizer(cfg["model"]["backbone"]); checkpoint=Path(args.checkpoint_dir) if args.checkpoint_dir else Path(cfg["training"]["output_dir"])/"best_model"; base=load_qwen3_4b(cfg["model"]["backbone"],len(tokenizer),False); llm=PeftModel.from_pretrained(base,checkpoint/"lora"); s=data["train"][0]; mc=cfg["model"]; model=CausalERCDialogueGCNQwen3(llm,token_ids,len(s["text_feats"][0]),len(s["audio_feats"][0]),len(s["visual_feats"][0]),hidden_dim=mc["hidden_dim"],num_classes=mc["num_classes"],heads=mc["attention_heads"],dropout=mc["dropout"],lambda_hgr=cfg["loss"]["lambda_hgr"],gcn_layers=mc["gcn_layers"],context_window=mc["graph_context_window"],history_window=mc["prompt_history_window"]).cuda(); load_non_llm(model,checkpoint/"non_llm.pt"); metrics,y,pred=evaluate_dialogues(model,data["test"],tokenizer,token_ids,cfg); result={**metrics,"labels":LABELS,"predictions":pred,"targets":y,"classification_report":report(y,pred,LABELS)}; output=Path(cfg["training"]["output_dir"])/"prediction_result.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
