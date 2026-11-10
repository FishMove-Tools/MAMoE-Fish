import bootstrap
import numpy as np
from capacity_policy import load
from environment import Environment
TASKS=['cruise','escape','speed']
def episode(job):
    path,task,seed,uniform=job;model,_=load(path);env=Environment();obs=env.reset_task(seed,task);total=np.zeros(3)
    for _ in range(env.horizon):
        obs,r,_,done,_=env.step(model.act(obs,uniform));total+=r
        if done:break
    return {**env.summary(),'episode_return':float(total.mean())}
def validation(pool,path,task,source_count=5):
    return pool.map(episode,[(str(path),TASKS.index(task),s,False) for s in range(860000,860000+source_count)])
def selection_score(rows,task):
    scale={'cruise':.12,'escape':60.,'speed':.012}[task]
    return 100*sum(r['success'] for r in rows)-float(np.mean([r['primary']/scale for r in rows]))
