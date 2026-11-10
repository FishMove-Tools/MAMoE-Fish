import bootstrap
from bootstrap import HERE,ROOT
import json,hashlib,time
import numpy as np
import torch

def clusters(k):
    out=ROOT/'development/initialization'; out.mkdir(exist_ok=True)
    file=out/f'routing_k{k}.npz'; teacher=ROOT/'development/teacher_data.npz'
    digest=hashlib.sha256(teacher.read_bytes()).hexdigest()
    if file.exists():
        z=np.load(file)
        assert str(z['teacher_sha256'])==digest
        return z['route']
    x=np.load(teacher)['observations'];speed=np.linalg.norm(x[:,7:9],axis=1)
    edges=np.quantile(speed,np.linspace(0,1,k+1))
    labels=np.searchsorted(edges[1:-1],speed,side='right')
    route=.95*np.eye(k)[labels]+.05/k
    assert np.isfinite(route).all() and np.allclose(route.sum(-1),1)
    np.savez_compressed(file,route=route.astype(np.float32),teacher_sha256=digest,speed_edges_mps=edges/20)
    (out/f'routing_k{k}.json').write_text(json.dumps({'experts':k,'mean_usage':route.mean(0).tolist(),'counts':np.bincount(labels,minlength=k).tolist(),'teacher_sha256':digest,'semantics':'source desired-speed quantiles; learned expert behavior requires interventions','speed_edges_mps':(edges/20).tolist()},indent=2))
    return route.astype(np.float32)

def warm_start(model,seed,steps=2000):
    start=time.perf_counter(); data=np.load(ROOT/'development/teacher_data.npz')
    x=torch.tensor(data['observations']); y=torch.tensor(data['actions'],dtype=torch.long)
    route=torch.tensor(clusters(model.arch.experts)) if model.arch.kind=='moe' else None
    prior=route.mean(0).tolist() if route is not None else []
    g=torch.Generator().manual_seed(seed+456)
    opt=torch.optim.Adam(model.actor.parameters(),lr=.0008)
    for step in range(steps):
        ids=torch.randint(len(y),(512,),generator=g); xb=x[ids]; yb=y[ids]
        if model.arch.kind=='moe':
            p,w,e=model.actor.parts(xb)
            ce=torch.nn.functional.nll_loss(p.clamp_min(1e-8).log(),yb)
            expert_nll=-e.clamp_min(1e-8).log().gather(-1,yb[:,None,None].expand(-1,model.arch.experts,1)).squeeze(-1)
            route_ce=-(route[ids]*model.actor.gate(xb).log_softmax(-1)).sum(-1).mean()
            loss=.5*ce+.5*(route[ids]*expert_nll).sum(-1).mean()+.05*route_ce+model.actor.regularizer(xb,prior)
        else: loss=torch.nn.functional.cross_entropy(model.actor(xb),yb)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.); opt.step()
    with torch.no_grad():
        sample=torch.arange(min(len(y),12000)); accuracy=float((model.actor(x[sample]).argmax(-1)==y[sample]).float().mean())
    return {'steps':steps,'examples_per_step':512,'loss':float(loss.detach()) if steps else None,'training_accuracy':accuracy,'elapsed_s':time.perf_counter()-start,'prior':prior}

