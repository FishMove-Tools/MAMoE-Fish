from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('rollout',type=Path);a=p.parse_args()
z=np.load(a.rollout,allow_pickle=False)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for i,c in enumerate(['#537cab','#a591c9','#57aeb5']):
    ax.plot(z['q'][:,i,0],z['q'][:,i,1],color=c,label=f'Fish {i}')
    ax.scatter(z['q'][-1,i,0],z['q'][-1,i,1],color=c,s=30)
ax.set(xlabel='x (m)',ylabel='y (m)',title='Recorded planar trajectory');ax.set_aspect('equal',adjustable='datalim')
ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
out=a.rollout.with_suffix('.png');fig.savefig(out,dpi=200);print(out)
