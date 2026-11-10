"""Source-only teacher initialization: cruise episodes and local tracking labels."""
import bootstrap
from bootstrap import ROOT
import multiprocessing as mp
import numpy as np, json, time, hashlib
from environment import Environment
_ENV=None

def teacher(x):
    err=x[:,:2]/4;vel=x[:,2:4]/20;ref=x[:,7:9]/20
    desired=ref+.18*err
    speed=np.clip(ref[:,0]+.25*err[:,0],0,.044)
    desired[:,0]=np.maximum(desired[:,0],.008)
    slip=np.arctan2(vel[:,1],np.maximum(vel[:,0],.015))
    angle=np.arctan2(desired[:,1],desired[:,0])-.2*np.clip(slip,-.3,.3)
    a=np.where(speed>.036,0,1)
    a=np.where((vel[:,0]>speed+.003)|(speed<.004),2,a)
    correction=angle-.7*x[:,4]/3
    return np.where(correction>.12,3,np.where(correction<-.12,4,a)).astype(np.int64)

def episode(seed):
    global _ENV
    if _ENV is None:_ENV=Environment()
    env=_ENV;o=env.reset_source(seed)
    rng=np.random.default_rng(seed+987);xs=[];ys=[]
    for _ in range(env.horizon):
        assert env.task==0 and np.all(o[:,32]==1) and np.all(o[:,29:32]==0)
        a=teacher(o);xs.extend(o.copy());ys.extend(a.copy())
        noisy=a.copy();mask=rng.random(3)<.10;noisy[mask]=rng.integers(0,5,mask.sum())
        o,_,_,done,_=env.step(noisy)
        if done:break
    return np.array(xs),np.array(ys),env.summary()

def main():
    start=time.perf_counter();seeds=list(range(100000,100180))
    with mp.get_context('spawn').Pool(3) as pool:results=pool.map(episode,seeds)
    x=np.concatenate([r[0] for r in results]);y=np.concatenate([r[1] for r in results])
    dest=ROOT/'development/teacher_data.npz'
    assert not dest.exists(),'Do not overwrite a frozen teacher dataset'
    np.savez_compressed(dest,observations=x,actions=y)
    manifest={'task':'cruise','task_id':0,'seeds':seeds,'samples':len(y),'class_counts':np.bincount(y,minlength=5).tolist(),
      'source_only':True,'curriculum':'balanced constant-speed cruise regimes: .014-.022, .025-.034, .039-.041 m/s; low/mid start at rest, curvature +/-.6 and heading +/-1.2 rad; high starts near steady cruise (90-100% nominal speed), initial position error +/-20mm, heading +/-.12, curvature +/-.08; slot speed caps .038 low/mid or .041 high; no within-episode speed steps or threat','target_teacher_data':False,'perturbation_probability':.10,'source_interaction_steps':sum(r[2]['steps'] for r in results),
      'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'elapsed_s':time.perf_counter()-start,'episodes':[r[2] for r in results]}
    (ROOT/'development/teacher_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='episodes'}),flush=True)
if __name__=='__main__':main()
