import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d_model : int = 512, n_head : int = 8, dropout : float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def calc_attention(self, q, k, v, att_mask=None):
        scores = torch.matmul(q, k.transpose(2, 1)) /  math.sqrt(self.d_model)
        
        if att_mask is not None:
            if len(att_mask.shape) == 2:
                att_mask = att_mask.unsqueeze(0).repeat(scores.shape[0], 1, 1)
            elif att_mask.shape[0] == (scores.shape[0] // self.n_head):
                att_mask = att_mask.repeat_interleave(self.n_head, dim=0)
            scores = scores.masked_fill(att_mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v)
    
    def _split_heads(self, x):
        N, S, E = x.size()
        D = E // self.n_head
        return x.reshape(N, S, self.n_head, D)\
                .permute(0, 2, 1, 3)\
                .reshape(N * self.n_head, S, D)

    def _merge_heads(self, x):
        N, S, E = x.size()
        N //= self.n_head
        D = E * self.n_head
        return x.reshape(N, self.n_head, S, E)\
                .permute(0, 2, 1, 3)\
                .reshape(N, S, D)

    def forward(self, q, k, v, 
                att_mask : torch.Tensor = None,
                key_length_mask : torch.Tensor = None,
                query_length_mask : torch.Tensor = None):
        
        if query_length_mask is not None:
            q = q.masked_fill(query_length_mask[...,None] == 0., 0.)
        if key_length_mask is not None:
            k = k.masked_fill(key_length_mask[...,None] == 0., 0.)
            v = v.masked_fill(key_length_mask[...,None] == 0., 0.)

        q = self._split_heads(self.Q(q))
        k = self._split_heads(self.K(k))
        v = self._split_heads(self.V(v))
        scores = self.calc_attention(q, k, v, att_mask)
        scores = self._merge_heads(scores)
        return self.O(scores).contiguous()