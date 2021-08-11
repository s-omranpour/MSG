import torch
from torch import nn
from torch.nn import functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=config['n_head'], 
            dropout=config['dropout'], 
            batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config['d_model'], 
            num_heads=config['n_head'], 
            dropout=config['dropout'], 
            batch_first=True,
        )
        self.ff1 = nn.Linear(config['d_model'], config['d_inner'])
        self.ff2 = nn.Linear(config['d_inner'], config['d_model'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])
        self.norm_cross = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        self.activation = F.gelu
        
    
    def forward(self, 
                x : torch.Tensor, 
                x_length_mask : torch.Tensor = None, 
                self_att_mask : torch.Tensor = None,
                memories : torch.Tensor = None, 
                memories_length_mask : torch.Tensor = None, 
                cross_att_mask : torch.Tensor = None):
        
        ## self attention
        
        self_att, _ = self.self_attention(
            x, x, x, 
            key_padding_mask=x_length_mask, 
            attn_mask=self_att_mask,
            need_weights=False
        )   
        x = self.norm1(x + self_att)

        ## cross attention
        cross_att, _ = self.cross_attention(
            x, memories, memories,
            attn_mask=cross_att_mask,
            key_padding_mask=memories_length_mask,
            need_weights=False
        )
        x = self.norm_cross(x + cross_att)

        ## Feed Forward
        h = self.dropout(self.activation(self.ff1(x)))
        h = self.dropout(self.ff2(h))
        return self.norm2(x+h)
    
    
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.get('share_weights', False):
            layers = [
                TransformerDecoderLayer(config)
            ] * config['n_layer']
        else:
            layers = [
                TransformerDecoderLayer(config)
                for _ in range(config['n_layer'])
            ]
        self.layers = nn.ModuleList(layers)
        
    
    def forward(self, 
        x : torch.Tensor, 
        x_length_mask : torch.Tensor = None,
        memories : torch.Tensor = None, 
        memories_length_mask : torch.Tensor = None, 
        cross_att_mask : torch.Tensor = None):
        
        self_att_mask = self._generate_self_att_mask(x.shape[1]).to(x.device)
        cross_att_mask = cross_att_mask.unsqueeze(1).repeat(1,self.config['n_head'],1,1).view(-1, x.shape[1], memories.shape[1])
        
        for layer in self.layers:
            x = layer(x, x_length_mask, self_att_mask, memories, memories_length_mask, cross_att_mask)
        return x
    
    def _generate_self_att_mask(self, sz):
        return ~torch.tril(torch.ones(sz, sz)).bool()
