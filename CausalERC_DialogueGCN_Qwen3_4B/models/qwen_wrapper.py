from __future__ import annotations
import torch
from torch import nn

class Qwen3Wrapper(nn.Module):
    def __init__(self,llm,token_ids,hidden_dim,num_classes=6):
        super().__init__(); self.llm=llm; self.ids=token_ids; h=int(llm.config.hidden_size); self.audio_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.visual_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.text_projection=nn.Sequential(nn.Linear(hidden_dim,h),nn.LayerNorm(h)); self.norm=nn.LayerNorm(h); self.classifier=nn.Linear(h,num_classes)
    def forward(self,input_ids,attention_mask,audio,visual,text,positions):
        e=self.llm.get_input_embeddings()(input_ids).clone(); rows=torch.arange(input_ids.size(0),device=input_ids.device); e[rows,positions[0]]=self.audio_projection(audio).to(e.dtype); e[rows,positions[1]]=self.visual_projection(visual).to(e.dtype); e[rows,positions[2]]=self.text_projection(text).to(e.dtype); o=self.llm(inputs_embeds=e,attention_mask=attention_mask,output_hidden_states=True,return_dict=True,use_cache=False,logits_to_keep=1); last=attention_mask.sum(-1).long()-1; return self.classifier(self.norm(o.hidden_states[-1][rows,last]))

def load_qwen3_4b(model_name,tokenizer_size,train=True,lora_rank=16,lora_alpha=32,lora_dropout=0.05):
    if not torch.cuda.is_available(): raise RuntimeError("Qwen3-4B NF4 loading requires CUDA")
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True); base=AutoModelForCausalLM.from_pretrained(model_name,quantization_config=q,torch_dtype=dtype,trust_remote_code=True,device_map={"":0}); base.resize_token_embeddings(tokenizer_size); base.config.use_cache=False
    if train:
        from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False}); base=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=lora_rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout,bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    return base
