import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler
from .utils import sample_seed


class DistributedSampler(TorchDistributedSampler):
    def __init__(
        self, dataset, shuffle=False, seed=None, drop_last=False, num_replicas=None, rank=None
    ):
        if seed is None:
            seed = sample_seed()
        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
