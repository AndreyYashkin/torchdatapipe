import numpy as np
from itertools import permutations


class DefaultIndex2Color:
    def __init__(self):
        id2color = []
        id2color += list(set(permutations([0, 0, 255])))
        id2color += list(set(permutations([0, 255, 255])))
        id2color += list(set(permutations([128, 128, 255])))
        id2color += list(set(permutations([128, 128, 0])))
        # id2color.append([255, 255, 255])
        id2color = np.array(id2color, dtype=np.uint8)
        id2color = np.concatenate([id2color, id2color // 2])
        self.id2color = id2color

    def __len__(self):
        return len(self.id2color)

    def __getitem__(self, idx):
        idx = idx % len(self)
        return self.id2color[idx]
