import os
from typing import List
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from deepnote import MusicRepr, Constants

def get_track(file_path, const, window, src_instruments, trg_instruments):
    try:
        seq = MusicRepr.from_file(file_path, const=const)
        seq_insts = set(seq.get_instruments())
        ## seq should have at least 1 src instrument and 1 trg instrument
        if len(seq_insts.intersection(set(src_instruments))) > 0\
        and len(seq_insts.intersection(set(trg_instruments))) > 0\
        and len(seq_insts.intersection(set(src_instruments).union(set(trg_instruments)))) > 1:
            src_seq = seq.keep_instruments(src_instruments)
            trg_seq = seq.keep_instruments(trg_instruments)
            return src_seq.get_bars(), trg_seq.get_bars()
        return None
    except Exception as e:
        print(e)
        return None

class Acc2SoloDataset(Dataset):
    def __init__(
        self, 
        data_dir : str, 
        const : Constants = None, 
        trg_instruments : List[str] = ['piano'],
        src_instruments : List[str] = ['piano'],
        pad_value : int = 0,
        max_files : int = 100, 
        max_len : int = 1024,
        window_len : int = 10,
        n_jobs : int = 2):

        super().__init__()

        self.const = Constants() if const is None else const
        self.window_len = window_len
        self.max_len = max_len
        self.trg_instruments = trg_instruments
        self.src_instruments = src_instruments
        self.pad_value = pad_value

        ## loading midis
        files = list(filter(lambda x: x.endswith('.mid'), os.listdir(data_dir)))[:max_files]
        
        tracks = Parallel(n_jobs=n_jobs)(
            delayed(get_track)(
                data_dir + file, const, window_len, src_instruments, trg_instruments
            ) for file in tqdm(files)
        )
#         tracks = [get_track(data_dir + file, const, window_len, src_instruments, trg_instruments) for file in tqdm(files)]
        self.tracks = list(filter(lambda x: x is not None, tracks))
        lens = list(map(lambda x: len(x[0]), self.tracks))
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
        src_bars, trg_bars = self.tracks[ind]
        src_seq = MusicRepr.concatenate(src_bars[offset:offset + self.window_len])
        trg_seq = MusicRepr.concatenate(trg_bars[offset:offset + self.window_len])
        res = {}
        for trg_inst in trg_seq.get_instruments():
            if len(set(src_seq.get_instruments()).difference(set([trg_inst]))) > 0:
                src = src_seq.remove_instruments([trg_inst]).to_remi(ret='index') + [0]
                trg = trg_seq.keep_instruments([trg_inst]).to_remi(ret='index') + [0]
                if len(src) <= self.max_len and len(trg) <= self.max_len:
                    res[trg_inst] = {
                        'src' : src, 
                        'trg' : trg
                    }
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