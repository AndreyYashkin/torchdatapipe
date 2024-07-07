import math
import torch.distributed as dist
from .sampler import Sampler


class DistributedSampler(Sampler):
    def __init__(self, sampler, num_replicas=None, rank=None, drop_last=False):
        super().__init__([sampler])
        self.sampler = sampler
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
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas)
                / self.num_replicas  # \
                # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)  # \
            # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = list(self.sampler)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
