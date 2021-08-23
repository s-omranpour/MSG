from tqdm.notebook import tqdm
from deepmusic import MusicRepr, Constants
import torch
from torch import nn
import pytorch_lightning as pl

from src.modules import RemiEmbedding, RemiHead, TransformerMixDecoder, nucleus_sample

class EncoderMixDecoderPerformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        print('training tasks:', ', '.join(config['tasks']))
        
        self.embedding = RemiEmbedding(config['embedding'])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config['encoder']['d_model'], 
            nhead=config['encoder']['n_head'], 
            dim_feedforward=config['encoder']['d_inner'], 
            dropout=config['encoder']['dropout'], 
            activation='gelu', 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, config['encoder']['n_layer'])
        self.decoder = TransformerMixDecoder(config['decoder'])
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
    
    def calculate_loss(self, logits, labels, mask):
        mask = mask * (labels != -100).float()
        loss = self.criterion(logits.transpose(1,2), labels) * mask
        s = torch.sum(mask)
        if s > 0:
            return torch.sum(loss) / s
        return None
    
    def forward(self, task, trg_inst, inputs):
        assert task in self.config['tasks'], f"Model is not configured for {task}."
        return {
            's2s' : self.forward_s2s,
            'clm' : self.forward_clm,
            'mlm' : self.forward_mlm
        }[task](trg_inst, inputs)
    
    def forward_mlm(self, trg_inst, inputs):
        src = self.embedding(inputs[trg_inst]['X_masked'].long())
        src_length_mask = self._generate_length_mask(inputs[trg_inst])
        h = self.encoder(src, src_key_padding_mask=src_length_mask)
        logits = self.heads[trg_inst](h)
        loss = None
        if 'masked_labels' in inputs[trg_inst] and src_length_mask is not None:
            loss = self.calculate_loss(logits,  inputs[trg_inst]['masked_labels'].long(), src_length_mask)
        return logits, loss
    
    def forward_clm(self, trg_inst, inputs):
        trg = self.embedding(inputs[trg_inst]['X'].long())
        trg_length_mask = self._generate_length_mask(inputs[trg_inst])
        self_att_mask = self._generate_self_att_mask(trg.shape[1]).to(trg.device)
        h = self.decoder(trg, tgt_mask=self_att_mask, tgt_key_padding_mask=trg_length_mask)
        logits = self.heads[trg_inst](h)
        loss = None
        if 'labels' in inputs[trg_inst] and trg_length_mask is not None:
            loss = self.calculate_loss(logits,  inputs[trg_inst]['labels'].long(), trg_length_mask)
        return logits, loss
    
    def forward_s2s(self, trg_inst, inputs):
        assert trg_inst in self.heads
        
        ## making all masks
        self_att_mask = self._generate_self_att_mask(inputs[trg_inst]['X'].shape[1]).to(inputs[trg_inst]['X'].device)
        trg_length_mask = self._generate_length_mask(inputs[trg_inst])
        length_masks = {}
        cross_att_masks = {}
        for inst in inputs:
            if inst != trg_inst:
                length_masks[inst] = self._generate_length_mask(inputs[inst])
                cross_att_masks[inst] = self._generate_cross_att_mask(
                    src=inputs[inst]['X'], 
                    src_length_mask=length_masks[inst], 
                    trg=inputs[trg_inst]['X'], 
                    trg_length_mask=trg_length_mask
                )
        
        ## encoder
        memories = {}
        for inst in inputs:
            if inst != trg_inst:
                src = self.embedding(inputs[inst]['X'].long())
                memories[inst] = self.encoder(src, src_key_padding_mask=length_masks[inst])

        ## decoder
        trg = self.embedding(inputs[trg_inst]['X'].long())
        h = self.decoder(trg, memories, self_att_mask, cross_att_masks, trg_length_mask, length_masks)
        
        ## head
        logits = torch.nan_to_num(self.heads[trg_inst](h))
        loss = None
        if 'labels' in inputs[trg_inst]:
            loss = self.calculate_loss(logits,  inputs[trg_inst]['labels'].long(), trg_length_mask)
        return logits, loss
    
    def step(self, batch, mode='train'):
        losses = []
        for task in self.config['tasks']:
            for inst in self.heads:
                if inst in batch:
                    logits, loss = self.forward(task, inst, batch)
                    if loss is not None:
                        losses += [loss]
                        self.log(mode + '_' + task + '_' + str(inst), loss.item())
        
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
        
    def _generate_length_mask(self, inputs):
        if 'X_len' not in inputs:
            return None
        lengths = inputs['X_len']
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

        n_bars = seq.get_bar_count()
        tracks = seq.separate_tracks()
        for inst in tracks:
            tracks[inst] = tracks[inst].get_bars()
        
        res = []
        with torch.no_grad():
            for i in tqdm(range(n_bars)):
                s = max(0, i - window + 1)
                inputs = {}
                for inst in tracks:
                    src = MusicRepr.concatenate(tracks[inst][s:i+1]).to_remi(ret='index')
                    inputs[inst] = {'X' : torch.tensor(src).long().to(device).unsqueeze(0)}

                res_bar = [0]
                while True:
                    inputs[trg_inst] = {'X' : torch.tensor(res_bar).long().to(device). unsqueeze(0)}
                    logits, _ = self.forward_s2s(trg_inst, inputs)
                    next_tok = nucleus_sample(logits[0, -1, :].detach().cpu(), top_p=top_p, t=t)
                    if next_tok == 0:
                        break
                    res_bar += [next_tok]

                res += res_bar
        return res

   
    