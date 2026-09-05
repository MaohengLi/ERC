import pickle
from pathlib import Path
import numpy as np
LABELS=["angry","happy","sad","neutral","excited","frustrated"]
ALIASES={"angry":0,"anger":0,"ang":0,"happy":1,"hap":1,"happiness":1,"sad":2,"sadness":2,"neutral":3,"neu":3,"excited":4,"excitement":4,"exc":4,"frustrated":5,"frustration":5,"fru":5}
def label_index(x):
    i=int(x) if isinstance(x,(int,np.integer)) else ALIASES.get(str(x).lower(),-1)
    if not 0<=i<6: raise ValueError(f"unknown label {x!r}")
    return i
def load_iemocap(path):
    with Path(path).open("rb") as f: raw=pickle.load(f)
    out={}
    for split in ("train","dev","test"):
        out[split]=[]
        for s in raw[split]:
            n=len(s.text)
            if any(len(getattr(s,k))!=n for k in ("audio","visual","speaker","label","sentence")): raise ValueError(f"misaligned {s.vid}")
            out[split].append({"vid":str(s.vid),"utt_count":n,"text_feats":[np.asarray(x,dtype="float32") for x in s.text],"audio_feats":[np.asarray(x,dtype="float32") for x in s.audio],"visual_feats":[np.asarray(x,dtype="float32") for x in s.visual],"speakers":[str(x) for x in s.speaker],"labels":[label_index(x) for x in s.label],"texts":[str(x) for x in s.sentence]})
    return out
