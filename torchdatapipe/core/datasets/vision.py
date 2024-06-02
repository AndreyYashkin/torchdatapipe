import os
import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset, ConcatDataset
from torchdatapipe.types.vision import ImageScene
from torchdatapipe.utils.vision import rect_mode_size


# TODO должен быть от списка файлов
class ImageDataset(Dataset):
    def __init__(self, names, files, imgsz=None, rect_mode=False):
        self.names = names
        self.files = files
        self.imgsz = imgsz
        self.rect_mode = rect_mode

    @staticmethod
    def from_mask(root, mask, imgsz, transforms=[], recursive=False):
        names, images = [], []
        _images = sorted(glob(f"*{mask}", root_dir=root, recursive=recursive))
        images = []
        for image in _images:
            name = image[: -len(mask) + 1]
            names.append(name)
            image = os.path.join(root, image)
            images.append(image)

        return ImageDataset(names, images, imgsz, transforms)

    @staticmethod
    def from_dir(root, imgsz, transforms=[], recursive=False):
        exts = [".png", ".jpg", ".bmp"]
        # exts = ["color.png"]
        # mask = "**/*" if recursive else "*"
        mask = "*"
        datasets = []
        for e in exts:
            for ext in [e.lower(), e.upper()]:
                ds = ImageDataset.from_mask(root, mask + ext, imgsz, transforms, recursive)
                if len(ds):
                    print("ext", root, mask + ext, len(ds))
                if len(ds):
                    datasets.append(ds)
        return ConcatDataset(datasets)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.files[idx]
        image = cv2.imread(path)
        if self.imgsz is not None:
            imgsz = rect_mode_size(image.shape, self.imgsz) if self.rect_mode else self.imgsz
            image = cv2.resize(image, imgsz[::-1])

        # TODO убрать расширение
        return ImageScene(image=image, id=name)


class ReplacedBackgroundDataset(Dataset):
    def __init__(self, dataset, backrounds, p=1):
        self.dataset = dataset
        self.backrounds = backrounds
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx0, idx1 = idx
        item = self.dataset[idx0]

        if idx1 is None or np.random.rand() > self.p:
            return item

        back = self.backrounds[idx1]
        mask = self.get_backround_mask(item)

        item.image[mask > 0] = back.image[mask > 0]

        return item

    def get_backround_mask(self, item):
        raise NotImplementedError
