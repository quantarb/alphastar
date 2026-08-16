"""Race-conditioned 10-action macro policy with unit-composition decisions."""
import torch
from torch import nn
RACES=("Terran","Protoss","Zerg")
ACTIONS=("worker","supply","production","gas","tech","basic_army","ranged_army","advanced_army","expand","attack")
STATE_SIZE=17
class GeneralMacroPolicy(nn.Module):
 def __init__(self,width=224):
  super().__init__();self.race=nn.Embedding(3,16);self.net=nn.Sequential(nn.Linear(STATE_SIZE+16,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,len(ACTIONS)))
 def forward(self,state,race):return self.net(torch.cat((state,self.race(race)),-1))
