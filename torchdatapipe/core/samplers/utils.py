import numpy as np
import torch
import torch.distributed as dist


def sample_seed():
    if dist.is_available():
        assert dist.get_world_size() == 1
    sq = np.random.SeedSequence()
    return sq.generate_state(1)[0]


def split_seed(seed, num):
    sq = np.random.SeedSequence(entropy=seed)
    return sq.generate_state(num).tolist()


def shuffle_indices(indices, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=g).tolist()
    for idx in perm:
        yield indices[idx]
