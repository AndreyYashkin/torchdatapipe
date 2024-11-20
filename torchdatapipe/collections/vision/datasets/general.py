import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchdatapipe.core.utils.filesystem import find_mimetype_files
from torchdatapipe.collections.vision.types.vision import ImageScene


# TODO должен быть от списка файлов
class ImageDataset(Dataset):
    def __init__(self, names, files, transform=None):
        self.names = names
        self.files = files
        self.transform = transform

    @staticmethod
    def from_dir(root, transform=None, recursive=False):
        files = find_mimetype_files(root, "image", recursive)
        names = [os.path.basename(path) for path in files]
        files = [os.path.join(root, path) for path in files]
        return ImageDataset(names, files, transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.files[idx]
        assert os.path.isfile(path)
        image = cv2.imread(path)
        assert image is not None
        if self.transform is not None:
            image = self.transform(image)

        return ImageScene(image=image, id=name)


class ReplacedBackgroundDataset(Dataset):
    def __init__(self, dataset, backrounds, get_backround_mask_fn, p=1):
        self.dataset = dataset
        self.backrounds = backrounds
        self.get_backround_mask_fn = get_backround_mask_fn
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx0, idx1 = idx
        item = self.dataset[idx0]

        if idx1 is None or np.random.rand() > self.p:
            return item

        back = self.backrounds[idx1]
        mask = self.get_backround_mask_fn(item)

        item.image[mask > 0] = back.image[mask > 0]

        return item
