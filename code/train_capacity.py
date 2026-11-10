"""Shared source MAPPO / target adaptation runner; no hidden action corrections."""
import bootstrap
from bootstrap import HERE,ROOT,PREVIOUS
from capacity_policy import Architecture,Policy,ResidualActor
from initialization import warm_start
from evaluation import validation,selection_score
from dataclasses import asdict
import argparse,hashlib,json,multiprocessing as mp,random,time,zipfile
from pathlib import Path
import numpy as np
import torch

def save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.actor.named_parameters():
        if not p.requires_grad: h.update(n.encode()); h.update(p.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',choices=['moe','mlp','feedback_mlp'],required=True); ap.add_argument('--experts',type=int,default=3); ap.add_argument('--width',type=int,required=True)
    ap.add_argument('--seed',type=int,default=5101); ap.add_argument('--updates',type=int,default=250); ap.add_argument('--workers',type=int,default=3); ap.add_argument('--horizon',type=int,default=64)
    ap.add_argument('--stage',choices=['source','target','scratch'],default='source'); ap.add_argument('--adaptation',choices=['full','gate','adapter'],default='full')
    ap.add_argument('--source'); ap.add_argument('--uniform-init',action='store_true'); ap.add_argument('--lr',type=float,default=.00003); ap.add_argument('--warm-steps',type=int,default=2000)
    ap.add_argument('--eval-every',type=int,default=50); ap.add_argument('--name',default='pilot'); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--task',choices=['cruise','escape','speed'],required=True)
    args=ap.parse_args()
    if args.stage!='source':args.warm_steps=0
    if args.stage=='source':assert args.task=='cruise' and args.adaptation=='full' and not args.source
    if args.stage=='scratch':assert args.kind!='moe' and args.adaptation=='full' and not args.source
    arch=Architecture(args.kind,args.experts,args.width)
    tag=f'moe_k{args.experts}' if args.kind=='moe' else f'{args.kind}_w{args.width}'
    out=ROOT/'runs'/args.name/tag/f'seed_{args.seed}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not args.resume: raise RuntimeError('Existing run requires explicit resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model=Policy(arch)
    # Critic initialization is identical across actor architectures for each seed.
    with torch.random.fork_rng():
        torch.manual_seed(args.seed+17000)
        for m in model.critic.modules():
            if isinstance(m,torch.nn.Linear): m.reset_parameters()
    prior=[1/args.experts]*args.experts if args.kind=='moe' else []
    source_sha=None
    if args.stage=='target':
        assert args.source
        ck=torch.load(args.source,map_location='cpu',weights_only=False)
        source_config=json.loads((Path(args.source).parent/'configuration.json').read_text())
        assert source_config['teacher_sha256']==hashlib.sha256((ROOT/'development/teacher_data.npz').read_bytes()).hexdigest(), 'Source curriculum/data mismatch'
        assert ck['stage']=='source' and ck['args']['task']=='cruise', 'Targets must branch from a cruise source checkpoint'
        assert ck.get('contract')=='frozen_bank_feedback25_v17', 'Incompatible source version'
        assert ck['architecture']==asdict(arch)
        model.load_state_dict(ck['model']); prior=ck['warm_start']['prior']
        source_sha=hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        # Reset to a common critic, so target comparisons do not inherit critic quality differences.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed+17000)
            for m in model.critic.modules():
                if isinstance(m,torch.nn.Linear): m.reset_parameters()
        if args.adaptation!='full':
            for p in model.actor.parameters(): p.requires_grad_(False)
        if args.adaptation=='gate':
            assert args.kind=='moe'
            for p in model.actor.gate.parameters(): p.requires_grad_(True)
        if args.uniform_init:
            assert args.kind=='moe'
            torch.nn.init.zeros_(model.actor.gate[-1].weight); torch.nn.init.zeros_(model.actor.gate[-1].bias)
        if args.adaptation=='adapter':
            assert args.kind in ['mlp','feedback_mlp']; model.actor=ResidualActor(model.actor)
        warm={'steps':0,'prior':prior,'source':str(args.source),'source_sha256':source_sha}
    critical=['bootstrap.py','train_capacity.py','capacity_policy.py','initialization.py','evaluation.py','protocol.json']
    identities={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in critical}
    for p in [HERE/n for n in ['environment.py','physics.py','worker.py']]+[ROOT/'group.xml',ROOT/'development/teacher_data.npz']:
        identities['src/'+p.name if p.parent==HERE else str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    identities['source'] = source_sha
    identity=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
    if not args.resume:
        with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as snapshot:
            for name in critical+['environment.py','physics.py','worker.py']:
                snapshot.write(HERE/name,'src/'+name)
            snapshot.write(ROOT/'group.xml','group.xml')
    start=time.perf_counter(); offset=0.; eval_s=0.; first=0; best=-1e9; restored=None
    active=[p for p in model.actor.parameters() if p.requires_grad]
    ao=torch.optim.Adam(active,lr=args.lr,eps=1e-5); co=torch.optim.Adam(model.critic.parameters(),lr=.0003,eps=1e-5)
    if args.resume:
        restored=torch.load(out/'latest.pt',weights_only=False)
        assert restored['identity']==identity,'Code changed; use a new experiment revision'
        for k,v in vars(args).items():
            if k not in ['updates','resume']: assert restored['args'][k]==v,k
        model.load_state_dict(restored['model']); ao.load_state_dict(restored['actor_optimizer']); co.load_state_dict(restored['critic_optimizer'])
        torch.set_rng_state(restored['torch_rng']); np.random.set_state(restored['numpy_rng']); random.setstate(restored['python_rng'])
        first=restored['update']; best=restored['best_score']; offset=restored['elapsed_s']; eval_s=restored['eval_s']; warm=restored['warm_start']; prior=warm['prior']
    elif args.stage=='source':
        warm=warm_start(model,args.seed,args.warm_steps); prior=warm['prior']
    else:
        warm={'steps':0,'prior':prior,'source':args.source,'source_sha256':source_sha,'target_teacher_steps':0,'elapsed_s':0.}
    frozen=frozen_hash(model)
    config={**vars(args),'architecture':asdict(arch),'actor_parameters':sum(p.numel() for p in model.actor.parameters()),'actor_trainable':sum(p.numel() for p in active),'critic_parameters':sum(p.numel() for p in model.critic.parameters()),'identity':identity,'critical_hashes':identities,'pcgrad':False,'task':args.task,'gamma':.99,'gae_lambda':.95,'epochs':4,'minibatch_team_steps':128,'teacher_sha256':hashlib.sha256((ROOT/'development/teacher_data.npz').read_bytes()).hexdigest(),'frozen_hash':frozen}
    (out/'configuration.json').write_text(json.dumps(config,indent=2)); (out/'COMPLETE.json').unlink(missing_ok=True)
    def pack(update): return {'model':model.state_dict(),'architecture':asdict(arch),'adaptation':args.adaptation,'stage':args.stage,'update':update,'args':vars(args),'identity':identity,'warm_start':warm,'contract':'frozen_bank_feedback25_v17'}
    ctx=mp.get_context('spawn'); pool=ctx.Pool(3); pipes=[]; processes=[]
    def validate(update):
        nonlocal best,eval_s
        t=time.perf_counter(); save(pack(update),out/'candidate.pt')
        rows=validation(pool,out/'candidate.pt',task=args.task); elapsed=time.perf_counter()-t; eval_s+=elapsed
        score=selection_score(rows,args.task)
        if score>best: best=score; save({**pack(update),'validation_score':score},out/'best.pt')
        record={'update':update,'team_steps':update*args.workers*args.horizon,'rows':rows,'score':score,'elapsed_s':offset+time.perf_counter()-start,'evaluation_s':elapsed}
        with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        brief={'tag':tag,'update':update,'stage':args.stage,'task':args.task,'adaptation':args.adaptation,
               'successes':sum(r['success'] for r in rows),'scenes':len(rows),'mean_primary':float(np.mean([r['primary'] for r in rows])),'score':score}
        print(json.dumps(brief),flush=True)
    try:
        if not args.resume:
            save(pack(0),out/'warm_start.pt'); validate(0)
        from worker import worker
        for i in range(args.workers):
            parent,child=ctx.Pipe(); wa=(child,args.seed*100+i,['cruise','escape','speed'].index(args.task))
            p=ctx.Process(target=worker,args=wa); p.start(); pipes.append(parent); processes.append(p)
        obs=np.stack([c.recv() for c in pipes])
        ep_returns=np.zeros((args.workers,3),dtype=np.float64);ep_lengths=np.zeros(args.workers,dtype=int);ep_count=0
        if restored:
            for c,s in zip(pipes,restored['environment_states']): c.send(('restore',s))
            obs=np.stack([c.recv() for c in pipes]);ep_returns=restored['episode_returns'];ep_lengths=restored['episode_lengths'];ep_count=restored['episode_count']
        for update in range(first,args.updates):
            rollout_start=time.perf_counter()
            O=[]; A=[]; LP=[]; V=[]; R=[]; D=[]; NV=[]
            for t in range(args.horizon):
                with torch.no_grad():
                    x=torch.tensor(obs); dist=model.distribution(x); a=dist.sample(); v=model.values(x)
                for c,act in zip(pipes,a.numpy()): c.send(act)
                newobs,reward,terminal,done,final,summary=zip(*[c.recv() for c in pipes])
                assert np.all(obs[:,0,32:35].argmax(-1)==['cruise','escape','speed'].index(args.task))
                ep_returns+=np.asarray(reward);ep_lengths+=1
                for wi,finished in enumerate(done):
                    if finished:
                        ep_count+=1
                        er={'episode':ep_count,'task':args.task,'seed':args.seed,'update':update+1,
                            'team_steps':update*args.workers*args.horizon+(t+1)*args.workers,'length':int(ep_lengths[wi]),
                            'return':float(ep_returns[wi].mean()),'fish_returns':ep_returns[wi].tolist(),'terminal':bool(terminal[wi]),'result':summary[wi]}
                        with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(er)+'\n')
                        ep_returns[wi]=0.;ep_lengths[wi]=0
                with torch.no_grad(): nv=model.values(torch.tensor(np.stack(final))).numpy()*(1-np.array(terminal)[:,None])
                O.append(obs); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy()); V.append(v.numpy()); R.append(reward); D.append(done); NV.append(nv); obs=np.stack(newobs)
            rollout_s=time.perf_counter()-rollout_start
            optimization_start=time.perf_counter()
            rv=np.array(R); vv=np.array(V); nv=np.array(NV); done=np.array(D); adv=np.zeros_like(rv); last=np.zeros_like(rv[0])
            for t in reversed(range(args.horizon)):
                last=rv[t]+.99*nv[t]-vv[t]+.99*.95*(1-done[t,:,None])*last; adv[t]=last
            returns=torch.tensor(adv+vv,dtype=torch.float32).flatten(0,1); adv=torch.tensor((adv-adv.mean())/(adv.std()+1e-8),dtype=torch.float32).flatten(0,1)
            ob=torch.tensor(np.array(O)).flatten(0,1); ac=torch.tensor(np.array(A)).flatten(0,1); oldlp=torch.tensor(np.array(LP)).flatten(0,1)
            cached=model.frozen_cache(ob)
            losses=[]
            for epoch in range(4):
                for ix in torch.randperm(len(ob)).split(128):
                    dist=model.distribution(ob[ix],None if cached is None else cached[ix]); lp=dist.log_prob(ac[ix]); ratio=(lp-oldlp[ix]).exp()
                    elements=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix])-.003*dist.entropy()
                    reg=model.actor.regularizer(ob[ix],prior) if args.stage=='source' else ob.new_tensor(0.)
                    loss=elements.mean()+reg; assert torch.isfinite(loss)
                    ao.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(active,.5); ao.step()
                    squared=.5*(model.values(ob[ix])-returns[ix]).square(); co.zero_grad()
                    squared.mean().backward()
                    torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5); co.step(); losses.append(float(loss.detach()))
            optimization_s=time.perf_counter()-optimization_start
            row={'update':update+1,'team_steps':(update+1)*args.workers*args.horizon,'reward':float(rv.mean()),'loss':float(np.mean(losses)),'elapsed_s':offset+time.perf_counter()-start,'eval_s':eval_s,'rollout_s':rollout_s,'optimization_s':optimization_s}
            assert frozen_hash(model)==frozen, 'Frozen policy parameters changed'
            with (out/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if (update+1)%args.eval_every==0 or update+1==args.updates or (out/'STOP').exists():
                assert frozen_hash(model)==frozen,'Frozen actor tensors changed'
                validate(update+1)
                for c in pipes: c.send('snapshot')
                states=[c.recv() for c in pipes]
                save({**pack(update+1),'actor_optimizer':ao.state_dict(),'critic_optimizer':co.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'environment_states':states,'episode_returns':ep_returns,'episode_lengths':ep_lengths,'episode_count':ep_count,'best_score':best,'elapsed_s':offset+time.perf_counter()-start,'eval_s':eval_s},out/'latest.pt')
                if (out/'STOP').exists():
                    print('Stopped at saved update boundary',flush=True);return
        (out/'COMPLETE.json').write_text(json.dumps({'updates':args.updates,'team_steps':args.updates*args.workers*args.horizon,'elapsed_s':offset+time.perf_counter()-start,'evaluation_s':eval_s,'acceptance_passed':False,'scope':'fixed-task training completed; acceptance requires independent evaluation','task':args.task,'episodes':ep_count,'frozen_hash_final':frozen_hash(model)},indent=2))
    finally:
        for c in pipes:
            try: c.send(None)
            except (EOFError,BrokenPipeError): pass
        for p in processes: p.join(20)
        pool.close(); pool.join()

if __name__=='__main__': main()
