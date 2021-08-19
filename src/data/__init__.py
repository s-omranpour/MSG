from torch.utils.data import DataLoader, random_split

from .acc2solo import Acc2SoloDataset
from .multi import MultiTrackDataset

def get_dataloaders(dataset, val_frac=0.1, batch_size=32, n_jobs=2):
    n = len(dataset)
    t = int(val_frac * n)
    td, vd = random_split(dataset, [n-t, t])
    tl = DataLoader(dataset=td, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=n_jobs, collate_fn=dataset.fn)
    vl = DataLoader(dataset=vd, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=n_jobs, collate_fn=dataset.fn)
    return tl, vl