import numpy as np
from environment import Environment
def worker(conn,seed,task):
    rng=np.random.default_rng(seed);env=Environment()
    def reset():
        scene=int(rng.integers(1000000,2000000000))
        return env.reset_source(scene) if task==0 else env.reset_task(scene,task)
    obs=reset();conn.send(obs)
    try:
        while True:
            msg=conn.recv()
            if msg is None:break
            if isinstance(msg,str) and msg=='snapshot':
                state={k:v for k,v in env.__dict__.items() if k not in ['model','data','commands','last_cycle_q']}
                conn.send({'environment':state,'qpos':env.data.qpos.copy(),'qvel':env.data.qvel.copy(),
                    'time':env.data.time,'warmstart':env.data.qacc_warmstart.copy(),'ctrl':env.data.ctrl.copy(),'rng':rng.bit_generator.state})
                continue
            if isinstance(msg,tuple) and msg[0]=='restore':
                import mujoco
                s=msg[1];env.__dict__.update(s['environment']);env.data.qpos[:]=s['qpos'];env.data.qvel[:]=s['qvel'];env.data.time=s['time'];env.data.ctrl[:]=s['ctrl']
                env.model.opt.density=1000*env.hydro_scale;mujoco.mj_forward(env.model,env.data);env.data.qacc_warmstart[:]=s['warmstart'];rng.bit_generator.state=s['rng'];obs=env.observe();conn.send(obs);continue
            final,reward,terminal,done,_=env.step(msg)
            summary=env.summary() if done else None
            obs=reset() if done else final
            conn.send((obs,reward,terminal,done,final,summary))
    finally:conn.close()
