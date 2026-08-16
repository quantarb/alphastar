"""Learned actor and friendly-target pointers over the shared entity snapshot."""
import torch
from torch import nn

class RepairPolicy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.entity=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.context=nn.Sequential(nn.Linear(width,width),nn.GELU())
        self.actor=nn.Linear(width,width,bias=False);self.target=nn.Linear(width,width,bias=False)
    def forward(self, entities, mask):
        e=self.entity(entities); z=e.masked_fill(mask.unsqueeze(-1),0).sum(1)/(~mask).sum(1,keepdim=True).clamp(min=1); z=self.context(z)
        return (torch.einsum('bd,bnd->bn',self.actor(z),e).masked_fill(mask,-1e9), torch.einsum('bd,bnd->bn',self.target(z),e).masked_fill(mask,-1e9))
