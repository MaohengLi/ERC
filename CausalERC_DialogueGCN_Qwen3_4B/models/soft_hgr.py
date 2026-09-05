import torch
def soft_hgr_loss(text,audio,visual):
    xs=[x.reshape(-1,x.size(-1)).float() for x in (text,audio,visual)]
    if xs[0].size(0)<2:return xs[0].new_zeros(())
    d=xs[0].size(-1); total=xs[0].new_zeros(())
    for i in range(3):
        for j in range(i+1,3):
            q=xs[i]-xs[i].mean(0,True); v=xs[j]-xs[j].mean(0,True); q=q/(q.std(0,unbiased=False,keepdim=True)+1e-6); v=v/(v.std(0,unbiased=False,keepdim=True)+1e-6); total-=((q*v).sum(-1).mean()/d-0.5*torch.trace((q.T@q/q.size(0))@(v.T@v/v.size(0)))/d)
    return total/3
