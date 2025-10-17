import torch, math
from typing import Dict, List, Tuple

def cosine_sim(a:torch.Tensor, b:torch.Tensor, eps=1e-8):
    na = a.norm(p=2) + eps
    nb = b.norm(p=2) + eps
    return (a @ b) / (na*nb)

def cosine_matrix(grads:Dict[str,torch.Tensor]):
    keys = list(grads.keys())
    k = len(keys)
    M = torch.zeros((k,k), dtype=torch.float32, device=next(iter(grads.values())).device)
    for i,ki in enumerate(keys):
        for j,kj in enumerate(keys):
            M[i,j] = cosine_sim(grads[ki], grads[kj])
    return keys, M

def project_nonconflicting(ga:torch.Tensor, gb:torch.Tensor):
    """If dot<0, remove the component of ga along gb (PCGrad-like)."""
    dot = ga @ gb
    if dot >= 0:
        return ga
    return ga - (dot / (gb.norm()**2 + 1e-8)) * gb

def conflict_fraction(M:torch.Tensor, tau:float=-0.1):
    k = M.shape[0]
    cnt = 0
    tot = k*(k-1)
    for i in range(k):
        for j in range(k):
            if i==j: continue
            if M[i,j] < tau:
                cnt += 1
    return cnt / max(1, tot)
