import torch
from torch import nn
    
class RemiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config['d_model'], config['n_vocab'])
        
    def forward(self, h):
        return self.head(h)