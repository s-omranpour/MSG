import math
import torch
from torch import nn
from deepmusic import Constants

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.long()) * math.sqrt(self.d_model)
    
    
class NotePositionalEmbedding(nn.Module):
    def __init__(self, d_emb, const, max_bar=10):
        super().__init__()
        self.const = const
        self.max_bar = max_bar
        self.pos_emb = Embeddings(const.n_bar_steps, d_emb)
        self.bar_emb = Embeddings(max_bar, d_emb)
        
    def get_pos(self, x):
        toks = [self.const.all_tokens[idx] for idx in x]
        poses = []
        for i, tok in enumerate(toks):
            if tok == 'Bar':
                poses += [0]
            elif tok.startswith('BeatPosition'):
                poses += [int(tok.split('_')[1])]
            else:
                poses += [poses[-1]]
        return torch.tensor(poses)
    
    def get_bar(self, x):
        toks = [self.const.all_tokens[idx] for idx in x]
        bars = [-1]
        for i, tok in enumerate(toks):
            if tok == 'Bar':
                v = min(bars[-1] + 1, self.max_bar - 1)
                bars += [v]
            else:
                bars += [bars[-1]]
        return torch.tensor(bars[1:])
        
    def forward(self, x):
        poses = torch.stack([self.get_pos(a) for a in x], dim=0).long().to(x.device)
        bars = torch.stack([self.get_bar(a) for a in x], dim=0).long().to(x.device)
        return self.pos_emb(poses) + self.bar_emb(bars)


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
        self.config = config
        
        self.tok_emb = Embeddings(config['n_vocab'], config['d_tok_emb'])

        assert config['positional_embedding'] in ['relative', 'note']
        if config['positional_embedding'] == 'relative':
            self.pos_emb = RelativePositionalEncoding(config['d_pos_emb'], config['max_len'])
        elif config['positional_embedding'] == 'note':
            self.pos_emb = NotePositionalEmbedding(config['d_pos_emb'], config['const'], config['max_bar'])

        d_in = config['d_tok_emb']
        if config['style_classes'] == 0:
            self.style_emb = None
        else:
            self.style_emb = Embeddings(config['style_classes'], config['d_style_emb'])
            d_in += config['d_style_emb']
        
        if config['concat_pos']:
            d_in += config['d_pos_emb']
        self.proj = nn.Linear(d_in, config['d_model'])
        self.dropout = nn.Dropout(p=config['dropout'])
        
    def forward(self, x, s=None):
        h = self.tok_emb(x.long())
        pos = self.pos_emb(x.long())
        if self.config['concat_pos']:
            h = torch.cat([h, pos], axis=-1)
        else:
            h += pos
        if self.style_emb is not None:
            style = self.style_emb(s)
            h = torch.cat([h, style], axis=-1)

        return self.dropout(self.proj(h))