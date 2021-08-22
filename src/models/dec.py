from tqdm.notebook import tqdm
from deepmusic import MusicRepr, Constants
import torch
from torch import nn
import pytorch_lightning as pl

from src.modules import RemiEmbedding, RemiHead, TransformerDecoderLayer, nucleus_sample

class BasePerformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.embedding = RemiEmbedding(config['embedding'])
        layer = TransformerDecoderLayer(config['decoder'])
        self.decoder = nn.TransformerDecoder(layer, config['decoder']['n_layer'])
        self.heads = nn.ModuleDict(
            [
                [inst, RemiHead(self.config['head'])] 
                for inst in self.config['instruments']
            ]
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['max_epochs'], eta_min=0.)
        return [opt]#, [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, inst, src, trg, src_len=None, trg_len=None, labels=None):
        assert inst in self.heads
        
        ## make all masks
        src_length_mask = self._generate_length_mask(src_len)
        trg_length_mask = self._generate_length_mask(trg_len)
        self_att_mask = self._generate_self_att_mask(trg.shape[1]).to(trg.device)
        cross_att_mask = self._generate_cross_att_mask(src, src_length_mask, trg, trg_length_mask)
        
        ## embedding
        trg = self.embedding(trg.long())
        src = self.embedding(src.long())

        ## decoder
        h = self.decoder(trg, src, self_att_mask, cross_att_mask, trg_length_mask, src_length_mask)
        
        ## head
        logits = torch.nan_to_num(self.heads[inst](h))
        loss = None
        if labels is not None:
            loss = self.calculate_loss(logits, labels.long(), trg_length_mask)
        return logits, loss
    
    def calculate_loss(self, logits, labels, mask):
        loss = self.criterion(logits.transpose(1,2), labels) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
    
    def step(self, batch, mode='train'):
        losses = []
        for inst in batch:
            logits, loss = self.forward(
                inst, 
                batch[inst]['src'], 
                batch[inst]['trg'], 
                batch[inst]['src_len'], 
                batch[inst]['trg_len'], 
                batch[inst]['labels']
            )
            losses += [loss]
            self.log(mode + '_' + str(inst), loss.item())
        
        if len(losses):
            total_loss = sum(losses) / len(losses)
            self.log(mode + '_loss', total_loss.item())
            return total_loss
        return None
        
    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')
        
    def _generate_self_att_mask(self, sz):
        return torch.tril(torch.ones(sz, sz))
        
    def _generate_length_mask(self, lengths):
        if lengths is None:
            return None
        mask = torch.zeros(len(lengths), max(lengths)).to(lengths.device)
        for i,l in enumerate(lengths):
            mask[i, :l] = 1.
        return mask
    
    def _generate_cross_att_mask(self, src, src_length_mask, trg, trg_length_mask):
        if src_length_mask is None or trg_length_mask is None:
            return None
        
        cross_att_mask = torch.zeros(src.shape[0], trg.shape[1], src.shape[1]).to(src.device)
        src_bar_mask = (src == 0) * src_length_mask
        trg_bar_mask = (trg == 0) * trg_length_mask

        for i in range(src.shape[0]):
            src_bars = torch.where(src_bar_mask[i] == 1.)[0].tolist() + [sum(src_length_mask[i]).int().item()]
            trg_bars = torch.where(trg_bar_mask[i] == 1.)[0].tolist() + [sum(trg_length_mask[i]).int().item()]
            assert len(trg_bars) == len(src_bars), f"trg_bars={trg_bars} length is not equal to src_bars={src_bars} length"
            n_bars = len(src_bars)

            ## we added an excess bar at the end so we just iterate on the first n-1 bars
            for j in range(n_bars-1):       
                s_trg = trg_bars[j]         ## trg j_th bar start
                e_trg = trg_bars[j+1]       ## trg j_th bar end
                e_src = src_bars[j+1]       ## src j_th bar end
                cross_att_mask[i, s_trg:e_trg, :e_src] = 1.
        return cross_att_mask
    
    
    def generate(self, trg_inst, seq, window=1, top_p=1., t=1., device='cuda'):
        self.eval()
        self.to(device)

        bars = seq.get_bars()
        n_bars = len(bars)
        res = []
        with torch.no_grad():
            for i in tqdm(range(n_bars)):
                s = max(0, i - window + 1)
                src = MusicRepr.concatenate(bars[s:i+1]).to_remi(ret='index')
                src = torch.tensor(src).long().to(device).unsqueeze(0)

                res_bar = [0]
                while True:
                    trg = torch.tensor(res_bar).long().to(device). unsqueeze(0)
                    logits, _ = self.forward(trg_inst, src, trg)
                    next_tok = nucleus_sample(logits[0, -1, :].detach().cpu(), top_p=top_p, t=t)
                    if next_tok == 0:
                        break
                    res_bar += [next_tok]

                res += res_bar
        return res

   
    