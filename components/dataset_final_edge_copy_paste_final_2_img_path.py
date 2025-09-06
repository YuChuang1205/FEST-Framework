import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from components.edges import onehot_to_binary_edges, mask_to_onehot
from components.copy_paste import copy_paste
import cv2


class SirstDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, mode='None', cp_probability=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.image_height = self.image_width = image_size
        self.cp_probability = cp_probability

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = cv2.resize(image, (self.image_height, self.image_width))
        mask = cv2.resize(mask, (self.image_height, self.image_width))

        mask = (mask > 127.5).astype(float)
        if (self.mode=='train'):
            if (np.random.random() < self.cp_probability):
                mask[mask == 1] = 255
                random_value = np.random.random()
                copy_index = int(np.random.random() * len(img_path))

                image_copy_path = os.path.join(self.image_dir, self.images[copy_index])
                image_copy = np.array(Image.open(image_copy_path).convert("RGB"))
                image_copy = cv2.resize(image_copy, (self.image_height, self.image_width))

                mask_copy_path = os.path.join(self.mask_dir, self.images[copy_index])
                mask_copy = np.array(Image.open(mask_copy_path).convert("L"), dtype=np.float32)  # images.convert(‘L’)为灰度图像
                mask_copy = cv2.resize(mask_copy, (self.image_height, self.image_width))

                mask_copy = (mask_copy > 127.5).astype(float)
                mask_copy[mask_copy == 1] = 255

                mask_copy2 = mask_copy.astype(np.uint8)
                mask2 = mask.astype(np.uint8)
                mask, image = copy_paste(image_copy, mask_copy2, image, mask2, random_value)
                mask[mask == 255] = 1
            mask = mask.astype(float)
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            mask_2 = mask.numpy()
            mask_2 = mask_2.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            edge[1, :] = 0
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.int64)

            return image, mask, edge


        elif (self.mode == 'val'):
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            mask_2 = mask.numpy()
            mask_2 = mask_2.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            edge[1, :] = 0
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.int64)
            return image, mask, edge
