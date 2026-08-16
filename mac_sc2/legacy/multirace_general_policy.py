"""Shared state encoder with race-specific macro-action heads."""
import torch
from torch import nn
from general_macro_policy import ACTIONS, STATE_SIZE

RACES=("Terran","Protoss","Zerg")

class MultiRaceGeneralMacroPolicy(nn.Module):
 def __init__(self,width=224):
  super().__init__()
  self.shared=nn.Sequential(nn.Linear(STATE_SIZE,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
  self.heads=nn.ModuleDict({race:nn.Linear(width,len(ACTIONS)) for race in RACES})
 def forward(self,state,race):
  z=self.shared(state)
  all_logits=torch.stack([self.heads[name](z) for name in RACES],dim=1)
  return all_logits[torch.arange(state.shape[0],device=state.device),race]
