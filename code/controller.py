"""Frozen deployment-actor interface. No critic, teacher or online optimizer."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from capacity_policy import Actor,Architecture

TASKS=('cruise','escape','speed')

class FAMoEController(nn.Module):
    def __init__(self,bundle):
        super().__init__()
        if isinstance(bundle,(str,Path)):
            bundle=torch.load(bundle,map_location='cpu',weights_only=True)
        assert bundle['format']=='famoe_frozen_bank_three_gates_v1'
        self.arch=Architecture(**bundle['architecture']);self.actor=Actor(self.arch)
        self.actor.load_state_dict(bundle['source_actor'])
        import copy
        self.target_gates=nn.ModuleDict({t:copy.deepcopy(self.actor.gate) for t in TASKS[1:]})
        for task in TASKS[1:]:self.target_gates[task].load_state_dict(bundle['target_gates'][task])
        self.metadata=bundle['metadata'];self.eval()
        for p in self.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def probabilities(self,observations,task):
        if task not in TASKS:raise ValueError(f'Unknown task: {task}')
        x=torch.as_tensor(observations,dtype=torch.float32)
        if x.shape[-1]!=35 or not torch.isfinite(x).all():raise ValueError('Expected finite observations with35 features')
        expected=torch.nn.functional.one_hot(torch.tensor(TASKS.index(task)),3).to(x.dtype)
        if not torch.equal(x[...,32:35],expected.expand_as(x[...,32:35])):
            raise ValueError('Observation task code differs from selected gate')
        expert_probs=self.actor.expert_probabilities(x)
        gate=self.actor.gate if task=='cruise' else self.target_gates[task]
        weights=gate(x).softmax(-1)
        return (weights[...,None]*expert_probs).sum(-2),weights

    @torch.no_grad()
    def act(self,observations,task):
        # Call only at complete primitive boundaries (0.5 s in this version).
        return self.probabilities(observations,task)[0].argmax(-1).cpu().numpy()
