"""Run a bounded, headless rollout with a released actor."""
from pathlib import Path
import argparse, importlib, json, sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from capacity_policy import Actor, Architecture

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['cruise','escape','speed'],default='cruise')
    p.add_argument('--steps',type=int,default=4,help='Maximum complete 0.5-second decisions')
    p.add_argument('--scene',type=int,default=2026102070)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/demo.npz')
    a=p.parse_args()
    if a.steps<1:p.error('--steps must be positive')
    ck=torch.load(ROOT/'checkpoints'/f'{a.task}.pt',map_location='cpu',weights_only=True)
    actor=Actor(Architecture(**ck['architecture']));actor.load_state_dict(ck['actor']);actor.eval()
    for v in actor.parameters():v.requires_grad_(False)
    env=importlib.import_module(ck['environment']).Environment()
    obs=env.reset_task(a.scene,['cruise','escape','speed'].index(a.task))
    rows=[];acts=[];weights=[];probs=[];times=[]
    for _ in range(min(a.steps,env.horizon)):
        with torch.no_grad():p,g,_=actor.parts(torch.as_tensor(obs,dtype=torch.float32))
        action=p.argmax(-1).numpy()
        obs,_,_,done,_=env.step(action)
        rows.append(env.q.copy());acts.append(action);weights.append(g.numpy());probs.append(p.numpy());times.append(env.data.time)
        if done:break
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,time=times,q=rows,actions=acts,weights=weights,probabilities=probs)
    info={'task':a.task,'environment':ck['environment'],'scene':a.scene,'steps':len(rows),'physical_time_s':float(env.data.time),'episode_complete':bool(done),'summary':env.summary()}
    a.output.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='summary'}))
if __name__=='__main__':main()
