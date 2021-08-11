import os
from typing import List
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from joblib import Parallel, delayed
from deepnote import MusicRepr, Constants

def get_track(file_path, const, window, instruments):
    try:
        seq = MusicRepr.from_file(file_path, const=const)
        seq_insts = seq.get_instruments()
        if len(set(seq_insts).intersection(set(instruments))) > 0 and len(seq_insts) > 1:
            return seq.get_bars()
        return None
    except Exception as e:
        print(e)
        return None
    
def get_dataloaders(dataset, val_frac=0.1, batch_size=32, n_jobs=2):
    n = len(dataset)
    t = int(val_frac * n)
    td, vd = random_split(dataset, [n-t, t])
    tl = DataLoader(dataset=td, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=n_jobs, collate_fn=dataset.fn)
    vl = DataLoader(dataset=vd, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=n_jobs, collate_fn=dataset.fn)
    return tl, vl

class MidiDataset(Dataset):
    def __init__(
        self, 
        data_dir : str, 
        const : Constants = None, 
        instruments : List[str] = ['piano'],
        pad_value : int = 0,
        max_files : int = 100, 
        window_len : int = 10,
        n_jobs : int = 2):

        super().__init__()

        self.const = Constants() if const is None else const
        self.window_len = window_len
        self.instruments = instruments
        self.pad_value = pad_value

        ## loading midis
        files = sorted(
            list(
                filter(lambda x: x.endswith('.mid'), os.listdir(data_dir))
            )[:max_files], 
            key=lambda x: os.stat(data_dir + x).st_size, 
            reverse=True
        )
        
        tracks = Parallel(n_jobs=n_jobs)(
            delayed(get_track)(data_dir + file, const, window_len, instruments) for file in tqdm(files)
        )
        
        self.tracks = list(filter(lambda x: x is not None, tracks))
        lens = list(map(len, self.tracks))
        self.lens = [max(0, l - self.window_len) + 1 for l in lens]
        self.cum_lens = [0] + [sum(self.lens[:i+1]) for i in range(len(self.lens))]

    def __len__(self):
        return self.cum_lens[-1]

    def get_idx(self, idx):
        for i, cl in enumerate(self.cum_lens):
            if idx < cl:
                return i-1, idx - self.cum_lens[i-1]

    def __getitem__(self, idx):
        ind, offset = self.get_idx(idx)
        seq = MusicRepr.concatenate(self.tracks[ind][offset:offset + self.window_len])
        res = {}
        for inst in self.instruments:
            tracks = seq.separate_tracks()
            if inst in tracks:
                trg = tracks[inst].to_remi(ret='index') + [0]
                acc_tracks = dict([(x, tracks[x]) for x in tracks if x != inst])
                if len(acc_tracks):
                    src = MusicRepr.merge_tracks(acc_tracks).to_remi(ret='index') + [0]
                    res[inst] = {'src' : src, 'trg' : trg}
        return res


    def fn(self, batch):
        def pad(x, l):
            return np.pad(x, (0, l), constant_values=self.pad_value)
        
        X = {}
        for i,b in enumerate(batch):
            for inst in b:
                if inst not in X:
                    X[inst] = {'src' : [], 'trg' : []}
                X[inst]['src'] += [b[inst]['src']]
                X[inst]['trg'] += [b[inst]['trg']]
#                 else:
#                     X[inst]['src'] += [np.zeros(self.window_len + 1)]
#                     X[inst]['trg'] += [np.zeros(self.window_len + 1)]
        
        res = {}
        for inst in X:
            src_len = torch.tensor([len(x)-1 for x in X[inst]['src']])
            trg_len = torch.tensor([len(x)-1 for x in X[inst]['trg']])
            src_M = max(src_len)
            trg_M = max(trg_len)
            if trg_M > 0:
                res[inst] = {
                    'src': torch.tensor([pad(x[:-1], src_M - l) for x,l in zip(X[inst]['src'], src_len)]),
                    'trg': torch.tensor([pad(x[:-1], trg_M - l) for x,l in zip(X[inst]['trg'], trg_len)]),
                    'src_len': src_len,
                    'trg_len': trg_len,
                    'labels': torch.tensor([pad(x[1:], trg_M - l) for x,l in zip(X[inst]['trg'], trg_len)])
                }
        return res