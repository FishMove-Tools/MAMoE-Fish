"""Three independent articulated swimmers; local actors, shared prescribed reference.
No policy action is replaced. This task generator never changes fish states after reset.
"""
import numpy as np
import mujoco
from physics import FishPhysics,DECISION_DT
OBS_DIM=35
TASK_VERSION="direct_escape_v3_aligned"
TASKS=['cruise','escape','speed']
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def rotation(a):return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

def fit_formation(positions,offsets):
    """Closest proper rigid transform; supplies shape, never an escape direction."""
    center=positions.mean(0)
    u,s,vt=np.linalg.svd(offsets.T@(positions-center))
    if np.linalg.det(u@vt)<0:u[:,-1]*=-1
    transform=u@vt
    return center+offsets@transform,float(np.arctan2(transform[0,1],transform[0,0]))

class Environment(FishPhysics):
    def __init__(self,render_meshes=False):
        super().__init__(3,render_meshes=render_meshes)
        self.reset_task()
    def reset_task(self,seed=0,task=0,difficulty=1.,ood=False):
        super().reset();self.rng=np.random.default_rng(seed);rng=self.rng
        self.seed=int(seed);self.task=int(task);self.ood=ood
        self.horizon=round((90. if task==1 else 60.)/DECISION_DT)
        self.curvature=rng.uniform(-.18,.18)*(1.5 if ood else 1)
        self.base_speed=rng.uniform(.022,.033)
        self.speed_levels=rng.uniform([.014,.029,.018],[.020,.036,.026])
        if ood:self.speed_levels=np.array([.012,.038,.025])
        self.offsets=np.array([[.4,0],[-.2,.35],[-.2,-.35]])
        self.ref,self.heading,self.targets=self.reference_at(0.)
        self.target_speed=float(self.speed_levels[0]) if task==2 else self.base_speed
        self.target_velocity=(self.reference_at(DECISION_DT)[2]-self.targets)/DECISION_DT
        self.q[:,:2]=self.targets+rng.uniform(-.06,.06,(3,2))*difficulty
        self.q[:,2]=rng.uniform(-.25,.25,3)*difficulty
        self.modes=np.full(3,2,int);self.cycle_vel=np.zeros((3,2));self.cycle_yawrate=np.zeros(3)
        self.threat=np.zeros(2);self.threat_active=False;self.threat_radius=.70
        angle=rng.uniform(-np.pi,np.pi);radius=rng.uniform(.36,.44) if ood else rng.uniform(.24,.32)
        self.threat_offset=radius*np.array([np.cos(angle),np.sin(angle)])
        self.threat_drift=.0045 if ood else .003;self.threat_trigger=10.
        self.clear_since=None;self.escape_time=None;self.ever_reentered=False
        self.task_version=TASK_VERSION
        self.metrics=[];self.steps_done=0;self.failed=False;self.minimum_distance=float('inf')
        self.hydro_scale=float(rng.uniform(.75,1.25)) if ood else 1.
        self.model.opt.density=1000*self.hydro_scale
        mujoco.mj_forward(self.model,self.data)
        return self.observe()
    def reset_source(self,seed):
        # Task 0 only. Speed and curvature remain constant within each episode.
        self.reset_task(seed,0)
        regime=int(seed)%3
        self.base_speed=float(self.rng.uniform(*[(.014,.022),(.025,.034),(.039,.041)][regime]))
        self.curvature=float(self.rng.uniform(-.08,.08) if regime==2 else self.rng.uniform(-.6,.6))
        # Bound every slot's prescribed speed, including the outer fish on a curve.
        slot_gain=np.linalg.norm(np.c_[1-self.curvature*self.offsets[:,1],self.curvature*self.offsets[:,0]],axis=1).max()
        self.base_speed=min(self.base_speed,(.041 if regime==2 else .038)/slot_gain)
        self.q[:,2]=self.rng.uniform(-.25,.25,3) if regime==2 else self.rng.uniform(-1.2,1.2,3)
        self.ref,self.heading,self.targets=self.reference_at(0.)
        self.target_speed=self.base_speed
        self.target_velocity=(self.reference_at(DECISION_DT)[2]-self.targets)/DECISION_DT
        if regime==2:
            # Fast cruise is initialized near steady swimming to cover small-error
            # high-speed states, rather than only acceleration lag from rest.
            self.q[:,:2]=self.targets+self.rng.uniform(-.02,.02,(3,2))
            self.q[:,2]=self.rng.uniform(-.12,.12,3)
            speed0=self.base_speed*self.rng.uniform(.9,1.,3)
            self.v[:,:2]=speed0[:,None]*np.c_[np.cos(self.q[:,2]),np.sin(self.q[:,2])]
            self.cycle_vel=self.v[:,:2].copy()
        mujoco.mj_forward(self.model,self.data)
        return self.observe()
    def reference_at(self,t):
        """Prescribed path indexed by time; no fish-state or threat input."""
        if self.task==2:
            arc=self.speed_levels@np.array([min(t,20.),np.clip(t-20.,0.,20.),max(0.,t-40.)])
        else:arc=self.base_speed*t
        angle=self.curvature*arc
        if abs(self.curvature)<1e-9:center=np.array([arc,0.])
        else:center=np.array([np.sin(angle),(1-np.cos(angle))])/self.curvature
        return center,float(angle),center+self.offsets@rotation(angle).T
    def update_reference(self):
        t=self.steps_done*DECISION_DT
        if self.task==2:self.target_speed=float(self.speed_levels[min(2,int(t/20))])
        else:self.target_speed=self.base_speed
        if self.task==1 and t>=self.threat_trigger:
            if not self.threat_active:
                self.threat=self.reference_at(self.threat_trigger)[0]+self.threat_offset
                self.threat_active=True
            self.ref=self.q[:,:2].mean(0)
            pursuit=self.ref-self.threat
            self.threat+=self.threat_drift*DECISION_DT*pursuit/max(np.linalg.norm(pursuit),1e-8)
            self.targets,self.heading=fit_formation(self.q[:,:2],self.offsets)
            self.target_velocity=np.zeros((3,2));self.target_speed=0.
            return
        self.ref,self.heading,self.targets=self.reference_at(t)
        self.target_velocity=(self.reference_at(t+DECISION_DT)[2]-self.targets)/DECISION_DT
    def observe(self):
        rows=[]
        for i in range(3):
            rot=rotation(self.q[i,2]);others=[j for j in range(3) if j!=i]
            err=(self.targets[i]-self.q[i,:2])@rot
            neighbors=((self.q[others,:2]-self.q[i,:2])@rot).ravel()
            nvel=((self.cycle_vel[others]-self.cycle_vel[i])@rot).ravel()
            threat=(self.threat-self.q[i,:2])@rot if self.threat_active else np.zeros(2)
            delta=wrap(self.heading-self.q[i,2])
            row=np.r_[err*4,(self.cycle_vel[i]@rot)*20,self.cycle_yawrate[i]*3,
                np.sin(delta),np.cos(delta),(self.target_velocity[i]@rot)*20,
                self.q[i,3:]/np.deg2rad([25,35]),self.v[i,3:]*.2,
                np.eye(5)[self.modes[i]],np.eye(3)[i],neighbors,nvel*20,
                threat,float(self.threat_active),np.eye(3)[self.task]]
            rows.append(row)
        obs=np.clip(np.array(rows,dtype=np.float32),-10,10)
        assert obs.shape==(3,OBS_DIM)
        return obs
    def step(self,actions,record=False):
        self.modes=np.asarray(actions,int)
        before=np.linalg.norm(self.targets-self.q[:,:2],axis=1)
        interval_reference_velocity=self.target_velocity.copy()
        old=self.q.copy();work=self.work.copy()
        prior_radius=np.linalg.norm(self.q[:,:2]-self.threat,axis=1) if self.threat_active else None
        trajectory=self.cycle(self.modes,record=record)
        self.cycle_vel=(self.q[:,:2]-old[:,:2])/DECISION_DT
        self.cycle_yawrate=(self.q[:,2]-old[:,2])/DECISION_DT
        self.steps_done+=1;self.update_reference()
        err=np.linalg.norm(self.targets-self.q[:,:2],axis=1)
        self.last_reference_velocity=interval_reference_velocity
        ve=np.linalg.norm(self.cycle_vel-interval_reference_velocity,axis=1)
        pairs=[(0,1),(0,2),(1,2)]
        distances=np.array([np.linalg.norm(self.last_cycle_q[:,i,:2]-self.last_cycle_q[:,j,:2],axis=1).min() for i,j in pairs])
        self.minimum_distance=min(self.minimum_distance,float(distances.min()))
        shape=np.array([(self.q[i,:2]-self.q[j,:2])-(self.targets[i]-self.targets[j]) for i,j in pairs])
        formation=float(np.sqrt(np.mean(np.sum(shape**2,axis=1))))
        if self.threat_active:
            rad=np.linalg.norm(self.q[:,:2]-self.threat,axis=1)
            unsafe=np.any(rad<=self.threat_radius+.18)
            reward=-.08*np.maximum(0,self.threat_radius+.18-rad)-3*formation**2-.005*(self.work-work)
            reward-=8*max(0,.47-distances.min())+.05*unsafe
            if prior_radius is not None:
                reward+=8*(rad-prior_radius)*np.any(prior_radius<self.threat_radius+.18)
            if np.all(rad>self.threat_radius+.18) and np.max([np.linalg.norm(self.q[i,:2]-self.q[j,:2]) for i,j in pairs])<1.2:
                if self.clear_since is None:self.clear_since=self.steps_done*DECISION_DT
                if self.steps_done*DECISION_DT-self.clear_since>=3 and self.escape_time is None:self.escape_time=self.clear_since-self.threat_trigger
            else:
                if self.escape_time is not None and unsafe:
                    if not self.ever_reentered:reward-=5
                    self.ever_reentered=True
                self.clear_since=None
        else:
            reward=10*(before-err)-2*err**2-65*ve**2-.05*wrap(self.heading-self.q[:,2])**2-.005*(self.work-work)
            reward-=.7*formation**2+3*max(0,.47-distances.min())
        terminal=bool(distances.min()<.40 or self.contact_force_peak>1e-5 or err.max()>.85)
        if terminal:reward-=10;self.failed=True
        reward=.7*reward+.3*reward.mean()
        self.metrics.append({'t':self.steps_done*DECISION_DT,'formation_m':formation,'path_m':float(np.sqrt(np.mean(err**2))),
                             'speed_mps':float(np.sqrt(np.mean(ve**2))),'min_distance_m':float(distances.min())})
        done=terminal or self.steps_done>=self.horizon
        return self.observe(),reward.astype(np.float32),terminal,done,trajectory
    def summary(self):
        m=[v for v in self.metrics if v['t']>=10] or self.metrics
        if not m:return {}
        formation=float(np.sqrt(np.mean([v['formation_m']**2 for v in m])))
        path=float(np.sqrt(np.mean([v['path_m']**2 for v in m])))
        speed=float(np.sqrt(np.mean([v['speed_mps']**2 for v in m])))
        complete=not self.failed and self.steps_done>=self.horizon and self.minimum_distance>=.40
        success=complete and formation<=.12 and path<=.15
        if self.task==1:success=complete and self.escape_time is not None and self.escape_time<=60 and formation<=.20 and not self.ever_reentered
        if self.task==2:success=success and speed<=.012
        primary=[formation,self.escape_time if self.escape_time is not None else 80.,speed][self.task]
        # Fixed penalties avoid omitting early failures from error comparisons.
        if not complete:primary=max(primary,[.85,80.,.08][self.task])
        return {'seed':self.seed,'task':TASKS[self.task],'task_version':self.task_version,'success':bool(success),'failed':self.failed,
                'formation_rmse_m':formation,'slot_rmse_m':path,'speed_rmse_mps':speed if self.task!=1 else None,'escape_time_s':self.escape_time,
                'reentered_danger':self.ever_reentered,'contact_violation':bool(self.contact_force_peak>1e-5),
                'min_separation_m':self.minimum_distance,'primary':float(primary),'steps':self.steps_done,
                'control_effort_J_proxy':self.work.tolist()}
