import os
from typing import List
import numpy as np
import random
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from deepmusic import MusicRepr, Constants

def get_track(file_path, const, window, instruments):
    try:
        seq = MusicRepr.from_file(file_path, const=const).keep_instruments(instruments)
        tracks = seq.separate_tracks()
        n_bar = seq.get_bar_count()
        if len(tracks.keys()) == len(instruments):
            return n_bar, dict(map(lambda inst: (inst,tracks[inst].get_bars()), tracks))
        return None
    except Exception as e:
        print(e)
        return None


class MultiTrackDataset(Dataset):
    def __init__(
        self, 
        data_dir : str, 
        files : list = None,
        const : Constants = None, 
        instruments : List[str] = ['piano'],
        pad_value : int = 0,
        max_files : int = 100, 
        max_len : int = 1024,
        window_len : int = 10,
        mask_prob : float = 0.15,
        n_jobs : int = 2):

        super().__init__()

        self.const = Constants() if const is None else const
        self.window_len = window_len
        self.max_len = max_len
        self.instruments = instruments
        self.pad_value = pad_value
        self.mask_prob = mask_prob

        ## loading midis
        if files is None:
            files = list(filter(lambda x: x.endswith('.mid'), os.listdir(data_dir)))[:max_files]
        else:
            files = files[:max_files]
        tracks = Parallel(n_jobs=n_jobs)(
            delayed(get_track)(
                data_dir + file, const, window_len, instruments
            ) for file in tqdm(sorted(files))
        )

        tracks = list(filter(lambda x: x is not None, tracks))
        self.tracks = [t[1] for t in tracks]
        self.lens = [max(0, t[0] - self.window_len) + 1 for t in tracks]
        self.cum_lens = [0] + [sum(self.lens[:i+1]) for i in range(len(self.lens))]

    def __len__(self):
        return self.cum_lens[-1]

    def get_idx(self, idx):
        for i, cl in enumerate(self.cum_lens):
            if idx < cl:
                return i-1, idx - self.cum_lens[i-1]

    def __getitem__(self, idx):
        def trim(tracks, offset, m, M):
            res = {}
            for inst in self.instruments:
                x = MusicRepr.concatenate(tracks[inst][offset:offset+self.window_len]).to_remi(ret='token')
                x = list(filter(lambda e: not e.startswith('NoteInstFamily'), x))
                x = [self.const.all_tokens.index(e) for e in x]
                res[inst] = x + [0]

            lens = [len(x) for x in res.values()]
            if sum(lens) < m or max(lens) > M:
                return {}
            return res
            
        ind, offset = self.get_idx(idx)
        tracks = self.tracks[ind]
        for l in range(self.window_len, 0, -1):
            res = trim(tracks, offset, m=100, M=self.max_len)
            if len(res):
                return res
        return {}
    
    def mask(self, tokens):
        res = []
        for i, tok in enumerate(tokens):
            prob = random.random()
            if prob <= self.mask_prob:
                res += [self.const.all_tokens.index('MASK')]
            else:
                res += [tok]
        return res


    def fn(self, batch):
        def pad(x, l):
            return np.pad(x, (0, l), constant_values=self.pad_value)
        
        X = dict([(inst, []) for inst in self.instruments])
        for b in batch:
            for inst in b:
                X[inst] += [b[inst]]
        
        res = {}
        for inst in X:
            if len(X[inst]):
                x_len = torch.tensor([len(x)-1 for x in X[inst]])
                M = max(x_len)
                res[inst] = {
#                     'X_masked' : torch.tensor([pad(self.mask(x[:-1]), Mx - l) for x,l in zip(X[inst], x_len)]),
                    'X': torch.tensor([pad(x[:-1], M - l) for x,l in zip(X[inst], x_len)]),
                    'X_len': x_len,
                    'labels': torch.tensor([pad(x[1:], M - l) for x,l in zip(X[inst], x_len)]),
                }
#                 res[inst]['masked_labels'] = res[inst]['X'].masked_fill(res[inst]['X'] == res[inst]['X_masked'], -100)
        return res