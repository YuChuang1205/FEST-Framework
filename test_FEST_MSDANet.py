#!/usr/bin/python3
"""
@Author : zhaojinmiao; yuchuang
@Time :
@desc:
"""
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import sys
from skimage import measure
from model.MSDA.MSDA_no_sigmoid import MSDANet_No_Sigmoid


def target_PD(copy_mask, target_mask):
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []
    un_overwrite_contours = []

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    for un_overwrite_c in un_overwrite_contours:
        temp_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(temp_contour_mask, [un_overwrite_c], (255))
        temp_mask = measure.label(temp_contour_mask, connectivity=2)
        coord_image = measure.regionprops(temp_mask)
        (y, x) = coord_image[0].centroid
        target_mask[int(y), int(x)] = 255

    return target_mask


def read_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    image_out_list = [line.strip() + '.png' for line in lines]
    return image_out_list


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


class SirstDataset(Dataset):
    def __init__(self, image_dir, img_list, transform_1=None, transform_2=None,
                 mode='None'):
        self.image_dir = image_dir
        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.image_list = img_list
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_list[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = image.shape

        if (self.mode == 'test'):
            if self.transform_1 is not None:
                augmentations_1 = self.transform_1(image=image)
                image_1 = augmentations_1["image"]
            if self.transform_2 is not None:
                augmentations_2 = self.transform_2(image=image)
                image_2 = augmentations_2["image"]
            return image_1, image_2, self.image_list[index], h, w

        else:
            print("mode输入的格式不对")
            sys.exit(0)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 8
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
root_path = os.path.abspath('')
input_path = os.path.join(root_path, 'images')
output_path = os.path.join(root_path, 'mask')
make_dir(output_path)
TEST_NUM = len(os.listdir(input_path))
txt_path = os.path.join(root_path, 'img_idx', 'test.txt')


def main():
    img_list = read_txt(txt_path)
    test_transforms_512 = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(
                mean=0.3426,
                std=0.1573,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    test_transforms_768 = A.Compose(
        [
            A.Resize(height=768, width=768),
            A.Normalize(
                mean=0.3426,
                std=0.1573,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_ds = SirstDataset(
        image_dir=input_path,
        img_list=img_list,
        transform_1=test_transforms_512,
        transform_2=test_transforms_768,
        mode='test'
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    model_1 = MSDANet_No_Sigmoid().to(DEVICE)
    model_1.load_state_dict(
        {k.replace('module.', ''): v for k, v in
         torch.load(r".\work_dirs\train_MSDANet\bestmIoUandPD_checkpoint_train_MSDANet.pth.tar"  # MSDANet_512.pth.tar
                    , map_location=DEVICE)['state_dict'].items()})
    model_1.eval()

    model_2 = MSDANet_No_Sigmoid().to(DEVICE)
    model_2.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load(r".\work_dirs\train_MSDANet_768\bestmIoUandPD_checkpoint_train_MSDANet_768.pth.tar",
                                        # MSDANet_512.pth.tar
                                        map_location=DEVICE)[
                                 'state_dict'].items()})
    model_2.eval()

    temp_num = 0

    for idx, (img_1, img_2, name, h, w) in enumerate(test_loader):
        print(idx)
        img_1 = img_1.to(device=DEVICE)
        img_2 = img_2.to(device=DEVICE)
        with torch.no_grad():
            output_1 = model_1(img_1)
            output_1 = torch.sigmoid(output_1)
            output_1 = output_1.cpu().data.numpy()

            output_2 = model_2(img_2)
            output_2 = torch.sigmoid(output_2)
            output_2 = output_2.cpu().data.numpy()

        for i in range(output_1.shape[0]):
            print(name[i])
            pred_1 = output_1[i]
            pred_1 = pred_1[0]
            pred_1 = np.array(pred_1, dtype='float32')
            pred_1 = cv2.resize(pred_1, (int(w[i]), int(h[i])))

            pred_2 = output_2[i]
            pred_2 = pred_2[0]
            pred_2 = np.array(pred_2, dtype='float32')
            pred_2 = cv2.resize(pred_2, (int(w[i]), int(h[i])))
            #

            pred = (pred_1 + pred_2) / 2
            pred_target = np.where(pred > 0.5, 255, 0)
            pred_target = np.array(pred_target, dtype='uint8')

            pred_copy_1 = np.where(pred_1 > 0.1, 255, 0)
            pred_copy_1 = np.array(pred_copy_1, dtype='uint8')
            pred_copy_2 = np.where(pred_2 > 0.1, 255, 0)
            pred_copy_2 = np.array(pred_copy_2, dtype='uint8')

            pred_out_1 = target_PD(pred_copy_1, pred_target)
            pred_out_2 = target_PD(pred_copy_2, pred_out_1)

            cv2.imwrite(os.path.join(output_path, name[i]), pred_out_2)


if __name__ == "__main__":
    main()
