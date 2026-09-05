from pathlib import Path
import json,torch
def save_checkpoint(model,tokenizer,directory,metadata):
    d=Path(directory); d.mkdir(parents=True,exist_ok=True); model.qwen.llm.save_pretrained(d/"lora"); tokenizer.save_pretrained(d/"tokenizer"); torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("qwen.llm.")},d/"non_llm.pt"); (d/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
def load_non_llm(model,path):
    missing,unexpected=model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False),strict=False); missing=[x for x in missing if not x.startswith("qwen.llm.")]
    if missing or unexpected: raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
