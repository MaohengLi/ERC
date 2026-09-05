import torch
from torch.utils.data import Dataset
from utils.prompt import build_prompt
from .iemocap import LABELS
class DialogueDataset(Dataset):
    def __init__(self,dialogues): self.dialogues=list(dialogues)
    def __len__(self): return len(self.dialogues)
    def __getitem__(self,i): return self.dialogues[i]
def _speaker_ids(xs):
    m={}; out=[]
    for x in xs: m.setdefault(x,len(m)); out.append(m[x])
    return out
def collate_dialogues(dialogues,tokenizer,token_ids,max_length=768,history_window=4,decisions=None):
    a_id,v_id,t_id=token_ids; lengths=torch.tensor([d["utt_count"] for d in dialogues]); B,T=len(dialogues),int(lengths.max()); td=len(dialogues[0]["text_feats"][0]); ad=len(dialogues[0]["audio_feats"][0]); vd=len(dialogues[0]["visual_feats"][0]); text=torch.zeros(B,T,td); audio=torch.zeros(B,T,ad); visual=torch.zeros(B,T,vd); spk=torch.zeros(B,T,dtype=torch.long); enc=[]; labels=[]; pos=[[],[],[]]; vids=[]; utts=[]
    for b,d in enumerate(dialogues):
        n=d["utt_count"]; text[b,:n]=torch.tensor(d["text_feats"]); audio[b,:n]=torch.tensor(d["audio_feats"]); visual[b,:n]=torch.tensor(d["visual_feats"]); spk[b,:n]=torch.tensor(_speaker_ids(d["speakers"]))
        for i in range(n):
            dec=None if decisions is None else decisions.get(d["vid"],[None]*n)[i]; p=build_prompt(d["speakers"],d["texts"],i,LABELS,history_window,dec)
            if hasattr(tokenizer,"apply_chat_template"): p=tokenizer.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
            x=tokenizer(p,truncation=True,max_length=max_length,return_tensors="pt")["input_ids"][0]
            for j,tok in enumerate((a_id,v_id,t_id)):
                where=(x==tok).nonzero(as_tuple=True)[0]
                if len(where)!=1: raise ValueError(f"special token count {len(where)} at {d['vid']}:{i}")
                pos[j].append(int(where[0]))
            enc.append(x); labels.append(d["labels"][i]); vids.append(d["vid"]); utts.append(i)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id; M=max(x.numel() for x in enc); ids=torch.full((len(enc),M),pad,dtype=torch.long); mask=torch.zeros_like(ids)
    for i,x in enumerate(enc): ids[i,:x.numel()]=x; mask[i,:x.numel()]=1
    return {"text":text,"audio":audio,"visual":visual,"speaker_ids":spk,"lengths":lengths,"input_ids":ids,"attention_mask":mask,"audio_positions":torch.tensor(pos[0]),"visual_positions":torch.tensor(pos[1]),"text_positions":torch.tensor(pos[2]),"labels":torch.tensor(labels),"vids":vids,"utt_indices":utts}
