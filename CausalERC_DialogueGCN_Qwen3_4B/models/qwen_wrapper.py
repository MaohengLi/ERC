from __future__ import annotations
import torch
from torch import nn

class Qwen3Wrapper(nn.Module):
    def __init__(self,llm,token_ids,hidden_dim,num_classes=6,micro_batch=1):
        super().__init__(); self.llm=llm; self.ids=token_ids; self.micro_batch=max(1,micro_batch); h=int(llm.config.hidden_size); self.audio_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.visual_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.text_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.norm=nn.LayerNorm(h); self.classifier=nn.Linear(h,num_classes)
    def forward(self,input_ids,attention_mask,audio,visual,text,positions):
        outputs=[]
        for start in range(0,input_ids.size(0),self.micro_batch):
            end=min(start+self.micro_batch,input_ids.size(0)); ids=input_ids[start:end]; mask=attention_mask[start:end]; rows=torch.arange(end-start,device=input_ids.device); e=self.llm.get_input_embeddings()(ids).clone(); e[rows,positions[0][start:end]]=self.audio_projection(audio[start:end]).to(e.dtype); e[rows,positions[1][start:end]]=self.visual_projection(visual[start:end]).to(e.dtype); e[rows,positions[2][start:end]]=self.text_projection(text[start:end]).to(e.dtype)
            base=self.llm.get_base_model() if hasattr(self.llm,"get_base_model") else self.llm; decoder=getattr(base,"model",None)
            if decoder is not None:
                o=decoder(inputs_embeds=e,attention_mask=mask,output_hidden_states=False,return_dict=True,use_cache=False); hidden=o.last_hidden_state
            else:
                o=self.llm(inputs_embeds=e,attention_mask=mask,output_hidden_states=True,return_dict=True,use_cache=False); hidden=getattr(o,"last_hidden_state",None); hidden=o.hidden_states[-1] if hidden is None else hidden
            last=mask.sum(-1).long()-1
            pooled=hidden[rows,last].to(self.norm.weight.dtype)
            outputs.append(self.classifier(self.norm(pooled)))
        return torch.cat(outputs,dim=0)

def load_qwen3_4b(model_name,tokenizer_size,train=True,lora_rank=16,lora_alpha=32,lora_dropout=0.05):
    if not torch.cuda.is_available(): raise RuntimeError("BF16 Qwen3-4B loading requires CUDA")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("Current GPU does not support BF16")
    from transformers import AutoModelForCausalLM
    dtype=torch.bfloat16; base=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype,trust_remote_code=True,device_map={"":0},low_cpu_mem_usage=True); base.resize_token_embeddings(tokenizer_size); base.config.use_cache=False
    if train:
        from peft import LoraConfig,TaskType,get_peft_model
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}); base.enable_input_require_grads(); base=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=lora_rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout,bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    return base
