import math
import torch
from torch import nn

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.long()) * math.sqrt(self.d_model)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :]

    
class RemiEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config['positional_embedding'] in ['none', 'relative', 'absolute']

        self.emb = nn.Embedding(config['n_vocab'], config['d_model'])
        if config['positional_embedding'] == 'none':
            self.pos_emb = None
        elif config['positional_embedding'] == 'relative':
            self.pos_emb = RelativePositionalEncoding(d_model=config['d_model'], max_len=config['max_len'])
        elif config['positional_embedding'] == 'absolute':
            self.pos_emb = nn.Embedding(config['max_len'], config['d_model'])
        self.dropout = nn.Dropout(p=config['dropout'])
        
    def forward(self, x):
        h = self.emb(x)
        if self.pos_emb is not None:
            pos = torch.arange(h.shape[1]).unsqueeze(0).repeat(h.shape[0], 1).to(h.device)
            h += self.pos_emb(pos)
        return self.dropout(h)