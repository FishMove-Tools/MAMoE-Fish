"""Probability-mixture actors with configurable expert count and matched capacity."""
import bootstrap
from dataclasses import dataclass, asdict
import torch
from torch import nn
from environment import OBS_DIM
torch.set_num_threads(1)

WIDTHS = {2:87, 3:68, 4:57, 6:44}

def mlp(n,w,out):
    return nn.Sequential(nn.Linear(n,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,out))

@dataclass
class Architecture:
    kind: str = 'moe'
    experts: int = 3
    width: int = 64
    gate_width: int = 32

class Actor(nn.Module):
    def __init__(self, arch):
        super().__init__(); self.arch = arch
        if arch.kind == 'moe':
            self.experts = nn.ModuleList([mlp(25, arch.width, 5) for _ in range(arch.experts)])
            self.gate = nn.Sequential(nn.Linear(OBS_DIM,arch.gate_width),nn.Tanh(),nn.Linear(arch.gate_width,arch.experts))
        else:
            self.net = mlp(25 if arch.kind=='feedback_mlp' else OBS_DIM,arch.width,5)
    def expert_probabilities(self,x):
        local=torch.cat([x[...,:5],x[...,9:29]],dim=-1)
        params=list(self.experts.parameters())
        if all(not p.requires_grad for p in params):
            # Pack only immutable experts. Version checks invalidate the cache on
            # state reload or any in-place modification; no extra learned weights.
            signature=tuple((p.data_ptr(),p._version,str(p.device),p.dtype) for p in params)
            if getattr(self,'_packed_signature',None)!=signature:
                self._packed_layers=[(torch.stack([e[i].weight for e in self.experts]).detach(),
                    torch.stack([e[i].bias for e in self.experts]).detach()[:,None,:]) for i in [0,2,4]]
                self._packed_signature=signature
            shape=local.shape[:-1]
            z=local.reshape(-1,25).unsqueeze(0).expand(self.arch.experts,-1,-1)
            for j,(w,b) in enumerate(self._packed_layers):
                z=torch.bmm(z,w.transpose(1,2))+b
                if j<2:z=z.tanh()
            return z.softmax(-1).permute(1,0,2).reshape(*shape,self.arch.experts,5)
        return torch.stack([m(local).softmax(-1) for m in self.experts],-2)
    def parts(self,x,uniform=False,cached=None):
        e = self.expert_probabilities(x) if cached is None else cached
        w = self.gate(x).softmax(-1)
        if uniform: w = torch.full_like(w,1/self.arch.experts)
        return (w.unsqueeze(-1)*e).sum(-2),w,e
    def forward(self,x,cached=None):
        if self.arch.kind=='moe':return self.parts(x,cached=cached)[0].clamp_min(1e-8).log()
        assert cached is None
        if self.arch.kind=='feedback_mlp':x=torch.cat([x[...,:5],x[...,9:29]],dim=-1)
        return self.net(x)
    def regularizer(self,x,prior):
        if self.arch.kind != 'moe': return x.new_tensor(0.)
        mean = self.gate(x).softmax(-1).flatten(0,-2).mean(0).clamp_min(1e-6)
        target = x.new_tensor(prior).clamp_min(1e-6)
        return .02*(mean*(mean.log()-target.log())).sum()

class Policy(nn.Module):
    def __init__(self,arch):
        super().__init__(); self.arch=arch; self.actor=Actor(arch)
        self.critic=mlp(3*OBS_DIM,128,3)
    def distribution(self,x,cached=None): return torch.distributions.Categorical(logits=self.actor(x,cached=cached))
    @torch.no_grad()
    def frozen_cache(self,x):
        if isinstance(self.actor,ResidualActor):
            assert all(not p.requires_grad for p in self.actor.base.parameters())
            return self.actor.base(x)
        if self.arch.kind=='moe' and all(not p.requires_grad for p in self.actor.experts.parameters()):
            return self.actor.expert_probabilities(x)
        return None
    def values(self,x): return self.critic(x.flatten(-2))
    @torch.no_grad()
    def act(self,obs,uniform=False):
        x=torch.as_tensor(obs,dtype=torch.float32)
        if uniform and self.arch.kind=='moe': return self.actor.parts(x,True)[0].argmax(-1).numpy()
        return self.actor(x).argmax(-1).numpy()

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Policy(Architecture(**ck['architecture']))
    if ck.get('adaptation')=='adapter': model.actor=ResidualActor(model.actor)
    model.load_state_dict(ck['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,ck

class ResidualActor(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base
        self.adapter=nn.Sequential(nn.Linear(OBS_DIM,30),nn.Tanh(),nn.Linear(30,5))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,x,cached=None): return (self.base(x) if cached is None else cached)+self.adapter(x)
    def regularizer(self,x,prior): return x.new_tensor(0.)

