from copy import deepcopy
from torch.utils.data import Dataset


class RAMCacheDataset(Dataset):
    def __init__(self, ds):
        self.items = [ds[idx] for idx in range(len(ds))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return deepcopy(self.items[idx])


class JoinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.__length = sum([len(ds) for ds in datasets])

    def __len__(self):
        return self.__length

    def __getitem__(self, idx):
        idx0, idx1 = idx
        return self.datasets[idx0][idx1]


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform_fn):
        self.ds = dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.transform_fn(item)
