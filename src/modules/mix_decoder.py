import torch
from torch import nn
from torch.nn import functional as F
from .att import Attention

class TransformerMixDecoderLayer(nn.Module):
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
                memories : dict[str, torch.Tensor], 
                tgt_mask : torch.Tensor = None, 
                memories_masks : dict[str, torch.Tensor] = None,
                tgt_key_padding_mask : torch.Tensor = None, 
                memories_key_padding_masks : dict[str, torch.Tensor] = None):
        
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

        ## mixed cross attention
        cross_atts = {}
        for inst in memories:
            if memories_masks is not None:
                memory_mask = memories_masks[inst].repeat_interleave(self.config['n_head'], dim=0)
            else:
                memory_mask = None
            
            if memories_key_padding_masks is not None:
                memory_key_padding_mask = memories_key_padding_masks[inst]
            else:
                memory_key_padding_mask = None
            
            cross_atts[inst] = self.cross_attention(
                x, memories[inst], memories[inst],
                att_mask=memory_mask,
                key_length_mask=memory_key_padding_mask,
                query_length_mask=tgt_key_padding_mask,
            )

        x = self.norm_cross(x + sum(cross_att.values))

        ## Feed Forward
        h = self.dropout(self.activation(self.ff1(x)))
        h = self.dropout(self.ff2(h))
        return self.norm2(x+h)
    
    
class TransformerMixDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerMixDecoderLayer(config) for _ in range(config['n_layer'])])
        
    def forward(self, 
                tgt : torch.Tensor, 
                memories : dict[str, torch.Tensor], 
                tgt_mask : torch.Tensor = None, 
                memories_masks : dict[str, torch.Tensor] = None,
                tgt_key_padding_mask : torch.Tensor = None, 
                memories_key_padding_masks : dict[str, torch.Tensor] = None):
        
        h = tgt
        for layer in self.layers:
            h = layer(h, memories, tgt_mask, memories_masks, tgt_key_padding_mask, memories_key_padding_masks)
        return h