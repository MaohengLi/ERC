from pathlib import Path
import json, shutil, torch
def save_checkpoint(model,tokenizer,directory,metadata):
    d=Path(directory); d.mkdir(parents=True,exist_ok=True); lora=d/"lora"
    if lora.exists(): shutil.rmtree(lora)
    model.qwen.llm.save_pretrained(lora,save_embedding_layers=False)
    tokenizer.save_pretrained(d/"tokenizer")
    emb=model.qwen.llm.get_input_embeddings().weight.detach().cpu(); ids=list(model.qwen.ids)
    torch.save({"token_ids":ids,"weights":emb[ids]},d/"feature_token_embeddings.pt")
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("qwen.llm.")},d/"non_llm.pt"); (d/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
def load_feature_token_embeddings(model,path):
    obj=torch.load(path,map_location="cpu",weights_only=False); ids=obj["token_ids"]; weights=obj["weights"]; emb=model.qwen.llm.get_input_embeddings().weight
    with torch.no_grad(): emb[ids].copy_(weights.to(emb.device,dtype=emb.dtype))
def load_non_llm(model,path):
    missing,unexpected=model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False),strict=False); missing=[x for x in missing if not x.startswith("qwen.llm.")]
    if missing or unexpected: raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
