import torch
from torch import nn
import pytorch_lightning as pl

from src.modules import RemiEmbedding, RemiHead, TransformerDecoder, nucleus_sample

class BasePerformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.embedding = RemiEmbedding(config['embedding'])
        self.decoder = TransformerDecoder(config['decoder'])
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
    
    def forward(self, inst, src, src_len, trg, trg_len, labels=None):
        assert inst in self.heads
        ## embedding
        x = self.embedding(trg.long())
        memories = self.embedding(src.long())
        
        ## make all masks
        x_length_mask = self._generate_length_mask(trg_len)
        memories_length_mask = self._generate_length_mask(src_len)
        cross_att_mask = self._generate_cross_att_mask(src, memories_length_mask, trg, x_length_mask)

        ## decoder
        h = self.decoder(x, x_length_mask, memories, memories_length_mask, cross_att_mask)
        
        ## head
        logits = torch.nan_to_num(self.heads[inst](h))
        loss = None
        if labels is not None:
            loss = self.calculate_loss(logits, labels.long(), (~x_length_mask).float())
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
                batch[inst]['src_len'], 
                batch[inst]['trg'], 
                batch[inst]['trg_len'], 
                batch[inst]['labels']
            )
            losses += [loss]
            self.log(mode + '_' + str(inst), loss.item())

        total_loss = sum(losses) / len(losses)
        self.log(mode + '_loss', total_loss.item())
        return total_loss
        
    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')
        
    def _generate_length_mask(self, lengths):
        if lengths is None:
            return None
        mask = torch.ones(len(lengths), max(lengths)).to(lengths.device).bool()
        for i,l in enumerate(lengths):
            mask[i, :l] = False
        return mask
    
    def _generate_cross_att_mask(self, src, src_length_mask, trg, trg_length_mask):
        if src_length_mask is None or trg_length_mask is None:
            return None
        
        cross_att_mask = torch.ones(src.shape[0], trg.shape[1], src.shape[1]).to(src.device).bool()
        
        src_bar_mask = (src == 0) * ~src_length_mask
        trg_bar_mask = (trg == 0) * ~trg_length_mask

        for i in range(src.shape[0]):
            src_bars = torch.where(src_bar_mask[i])[0].tolist() + [sum(~src_length_mask[i]).item()]
            trg_bars = torch.where(trg_bar_mask[i])[0].tolist() + [sum(~trg_length_mask[i]).item()]
            assert len(trg_bars) == len(src_bars), f"trg_bars={trg_bars} length is not equal to src_bars={src_bars} length"
            n_bars = len(src_bars)

            ## we added an excess bar at the end so we just iterate on the first n-1 bars
            for j in range(n_bars-1):       
                s_trg = trg_bars[j]         ## trg j_th bar start
                e_trg = trg_bars[j+1]       ## trg j_th bar end
                e_src = src_bars[j+1]       ## src j_th bar end
                cross_att_mask[i, s_trg:e_trg, :e_src] = False
        return cross_att_mask
    
    
    def generate(self, trg_inst, seq=None, window_len=1, max_len=1000, top_p=1., t=1.):
        self.eval()

        bars = seq.get_bars()
        n_bars = len(bars)

        with torch.no_grad():
            for i in tqdm(range(n_bars)):
                s = max(0, i - window_len + 1)
                x = np.array(MusicRepr.concatenate(bars[s:i+1]).to_remi(ret='index') + [0])
                inputs = {
                    trg_inst: {
                        'src' : torch.tensor(x).long().to(self.device).unsqueeze(0)
                    }
                }

                res_bar = [0]
                while True:
                    inputs[trg_inst]['trg'] = torch.tensor(res_bar).long().to(self.device). unsqueeze(0)
                    logits = self.forward_s2s(trg_inst=trg_inst, inputs=inputs)
                    next_tok = nucleus_sample(logits[0, -1, :].detach().cpu(), top_p=top_p, t=t)
                    if next_tok == 0:
                        break
                    res_bar += [next_tok]

                res += res_bar
        return np.array(res)

   
    