from torch.nn import functional as F


def nucleus_sample(x, top_p=1., t=1.):
    probs = F.softmax(x/t, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=0)
    threshold = cumsum_sorted_probs > top_p
    idx = probs.size(0) - sum(threshold) + 1
    cand_indices = sorted_indices[:idx]
    cand_probs = probs[cand_indices]
    cand_probs /= cand_probs.sum()
    word = np.random.choice(cand_indices.numpy(), size=1, p=cand_probs.numpy())[0]
    return word