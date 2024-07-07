import cv2
import numpy as np


def add_segmentation(image, segmentation, alpha=0.6):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.stack([gray_image] * 3, axis=-1)
    return cv2.addWeighted(image, alpha, segmentation, 1 - alpha, 0)


def rect_mode_size(old_imgsz, new_imgsz):
    if old_imgsz[0] / old_imgsz[1] > new_imgsz[0] / new_imgsz[1]:
        mul = new_imgsz[0] / old_imgsz[0]
        imgsz = (new_imgsz[0], int(old_imgsz[1] * mul))
    else:
        mul = new_imgsz[1] / old_imgsz[1]
        imgsz = (int(old_imgsz[0] * mul), new_imgsz[1])

    return imgsz


def cv2_image_collate_fn(images, norm=255):
    images = np.stack(images)
    images = np.moveaxis(images, [0, 1, 2, 3], [0, 2, 3, 1])
    images = np.flip(images, axis=1)  # BGR -> RGB
    return images / norm
