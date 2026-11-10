from pathlib import Path
import sys, numpy as np, torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'))
from physics import primitive, FishPhysics
from capacity_policy import Architecture, Actor
from environment import Environment
for a in range(5):
    assert np.allclose(primitive(a,0),0,atol=1e-12)
    assert np.allclose(primitive(a,.5),0,atol=1e-12)
env=Environment();o=env.reset_task(1,0)
assert o.shape==(3,35)
for task in ['cruise','escape','speed']:
    ck=torch.load(root/'checkpoints'/f'{task}.pt',map_location='cpu',weights_only=True)
    actor=Actor(Architecture(**ck['architecture']));actor.load_state_dict(ck['actor']);actor.eval()
    with torch.no_grad():p,w,e=actor.parts(torch.tensor(o))
    assert p.shape==(3,5) and torch.allclose(p.sum(-1),torch.ones(3),atol=1e-6)
    assert torch.allclose(w.sum(-1),torch.ones(3),atol=1e-6)
o,*_=env.step(np.array([0,1,2]));assert np.isfinite(o).all() and abs(env.data.time-.5)<1e-8
assert FishPhysics(render_meshes=True).model.nmesh==3
print('OK: primitive endpoints, checkpoint tensors, probability mixtures, one physics cycle, visual meshes')
