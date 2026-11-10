from pathlib import Path
import numpy as np
import mujoco
import mujoco.rollout
from bootstrap import ROOT
PHYSICS_DT=.0005
SERVO_DT=.0005
DECISION_DT=.5
PRIMITIVE_PERIOD=.5
ACTIVE_DURATION=.3
COAST_DURATION=.2
assert abs(DECISION_DT-PRIMITIVE_PERIOD)<1e-12, 'Current environment supports one full primitive per decision.'
assert abs(ACTIVE_DURATION+COAST_DURATION-PRIMITIVE_PERIOD)<1e-12
NAMES=['continuous','intermittent','glide','left','right']

def primitive(action,t):
    s=np.clip(t/PRIMITIVE_PERIOD,0,1)
    if action==2:return np.zeros(2)
    if action in [3,4]:
        def ease(u):
            u=np.clip(u,0,1)
            return u*u*u*(10+u*(-15+6*u))
        envelope=ease(t/.10) if t<.10 else 1-ease((t-.16)/.34)
        return np.deg2rad([24.,-33.])*envelope*(1 if action==3 else -1)
    u=min(t/ACTIVE_DURATION,1.) if action==1 else s
    env=np.sin(np.pi*u)**2
    phase=np.array([0.,-2.7])
    amps=np.deg2rad([30.,40.])*(.6 if action==1 else 1.)
    # Zero-mean phase basis; smooth envelope alone introduces a phase-dependent DC term.
    wave=amps*(np.sin(2*np.pi*u+phase)+.5*np.sin(phase))*env
    wave+=np.deg2rad([4.,4.] if action==1 else [-6.,-3.])*env
    if action==1 and t>=ACTIVE_DURATION:return np.zeros(2)
    return wave

class FishPhysics:
    def __init__(self,n=3,dt=PHYSICS_DT,render_meshes=False):
        self.n=n
        # Remove visual mesh assets for faster headless loading, retaining identical dynamics.
        source=ROOT/f'{"single" if n==1 else "group"}.xml'
        if render_meshes:self.model=mujoco.MjModel.from_xml_path(str(source))
        else:
            import xml.etree.ElementTree as E
            root=E.fromstring(source.read_text())
            for el in list(root.find('asset')):
                if el.tag=='mesh':root.find('asset').remove(el)
            for body in root.iter('body'):
                for geom in list(body):
                    if geom.tag=='geom' and geom.get('type')=='mesh':body.remove(geom)
            self.model=mujoco.MjModel.from_xml_string(E.tostring(root,encoding='unicode'))
        self.model.opt.timestep=dt
        self.data=mujoco.MjData(self.model)
        self.steps=round(PRIMITIVE_PERIOD/dt)
        assert abs(self.steps*dt-PRIMITIVE_PERIOD)<1e-10
        self.commands=np.stack([[primitive(a,(i+1)*dt) for a in range(5)] for i in range(self.steps)])
        self.reset()
    @property
    def q(self):return self.data.qpos.reshape(self.n,5)
    @property
    def v(self):return self.data.qvel.reshape(self.n,5)
    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.q[:,1]=np.arange(self.n)*.8
        mujoco.mj_forward(self.model,self.data)
        self.work=np.zeros(self.n);self.peak_torque=np.zeros((self.n,2));self.saturated=np.zeros((self.n,2),int)
        self.track_squared=np.zeros((self.n,2));self.ticks=0
    def cycle_slow(self,actions,record=False):
        actions=np.asarray(actions,int)
        assert actions.shape==(self.n,) and np.all((actions>=0)&(actions<5))
        records=[]
        for step in range(self.steps):
            ctrl=self.commands[step,actions]
            self.data.ctrl[:]=ctrl.ravel()
            mujoco.mj_step(self.model,self.data)
            tau=self.data.actuator_force.reshape(self.n,2)
            self.work+=np.sum(np.abs(tau*self.v[:,3:]),axis=1)*self.model.opt.timestep
            self.peak_torque=np.maximum(self.peak_torque,np.abs(tau))
            self.saturated+=(np.abs(tau)>.449)
            self.track_squared+=(self.q[:,3:]-ctrl)**2;self.ticks+=1
            if record:records.append(np.r_[self.data.time,self.q.ravel(),self.v.ravel(),ctrl.ravel(),tau.ravel()])
        if not np.isfinite(self.q).all():raise FloatingPointError('Non-finite articulated dynamics')
        return np.array(records) if record else None
    def cycle(self,actions,record=False):
        actions=np.asarray(actions,int)
        assert actions.shape==(self.n,) and np.all((actions>=0)&(actions<5))
        ctrl=self.commands[:,actions,:].reshape(self.steps,2*self.n)
        state=np.r_[self.data.time,self.data.qpos,self.data.qvel]
        states,forces=mujoco.rollout.rollout(self.model,self.data,state,ctrl[None],initial_warmstart=self.data.qacc_warmstart.copy(),persistent_pool=True)
        states=states[0];sensors=forces[0];forces=sensors[:,:self.n*2]
        self.contact_force_peak=float(np.max(sensors[:,self.n*2:]))
        q=states[:,1:1+5*self.n].reshape(self.steps,self.n,5)
        v=states[:,1+5*self.n:].reshape(self.steps,self.n,5)
        tau=forces.reshape(self.steps,self.n,2)
        self.work+=np.sum(np.abs(tau*v[:,:,3:]),axis=(0,2))*self.model.opt.timestep
        self.peak_torque=np.maximum(self.peak_torque,np.max(np.abs(tau),axis=0))
        self.saturated+=np.sum(np.abs(tau)>.449,axis=0)
        self.track_squared+=np.sum((q[:,:,3:]-ctrl.reshape(self.steps,self.n,2))**2,axis=0)
        self.ticks+=self.steps
        self.last_cycle_q=q
        if not np.isfinite(states).all() or states[-1,0]<state[0]+PRIMITIVE_PERIOD-1e-6:
            raise FloatingPointError('Divergent or reset physics rollout')
        return np.column_stack([states,ctrl,forces]) if record else None
