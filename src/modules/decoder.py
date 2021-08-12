import torch
from torch import nn
from torch.nn import functional as F
from .att import Attention

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attention = Attention(
            d_model=config['d_model'], 
            n_head=config['n_head'], 
            dropout=config['dropout']
        )
        self.cross_attention = Attention(
            d_model=config['d_model'], 
            n_head=config['n_head'], 
            dropout=config['dropout']
        )
        self.ff1 = nn.Linear(config['d_model'], config['d_inner'])
        self.ff2 = nn.Linear(config['d_inner'], config['d_model'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])
        self.norm_cross = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        self.activation = F.gelu
        
    
    def forward(self, 
                tgt : torch.Tensor, 
                memory : torch.Tensor, 
                tgt_mask : torch.Tensor = None, 
                memory_mask : torch.Tensor = None,
                tgt_key_padding_mask : torch.Tensor = None, 
                memory_key_padding_mask : torch.Tensor = None):
        
        ## self attention
        x = tgt
        if tgt_mask is None:
            tgt_mask = self._generate_self_att_mask(x.shape[1]).to(x.device)

        self_att = self.self_attention(
            x, x, x,
            att_mask=tgt_mask,
            key_length_mask=tgt_key_padding_mask, 
            query_length_mask=tgt_key_padding_mask
        )   
        x = self.norm1(x + self_att)

        ## cross attention
        if memory_mask is not None:
            memory_mask = memory_mask.repeat_interleave(self.config['n_head'], dim=0)
            
        cross_att = self.cross_attention(
            x, memory, memory,
            att_mask=memory_mask,
            key_length_mask=memory_key_padding_mask,
            query_length_mask=tgt_key_padding_mask,
        )
        x = self.norm_cross(x + cross_att)

        ## Feed Forward
        h = self.dropout(self.activation(self.ff1(x)))
        h = self.dropout(self.ff2(h))
        return self.norm2(x+h)