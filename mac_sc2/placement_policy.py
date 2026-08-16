import torch
from torch import nn
class PlacementPolicy(nn.Module):
 def __init__(self,abilities,width=128):
  super().__init__();self.entity=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width));self.ability=nn.Linear(width,abilities);self.candidate=nn.Sequential(nn.Linear(width+2,width),nn.GELU(),nn.Linear(width,1))
 def forward(self,entities,mask,candidates):
  h=self.entity(entities).masked_fill(mask[...,None],0); z=h.sum(1)/(~mask).sum(1,keepdim=True).clamp_min(1); n=candidates.shape[1]; return self.ability(z),self.candidate(torch.cat((z[:,None,:].expand(-1,n,-1),candidates),-1)).squeeze(-1)
