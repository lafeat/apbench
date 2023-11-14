import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
from tqdm import tqdm
import PIL
import os


def pixel_search(clean_train_data, pert_init=None, sparsity=1, dataset=None):
    H, W, C = (
        clean_train_data.data.shape[1],
        clean_train_data.data.shape[2],
        clean_train_data.data.shape[3],
    )
    pts = []

    if type(pert_init) in [np.ndarray]:
        perturbation = pert_init
    else:
        perturbation = np.zeros_like(clean_train_data.data, dtype=float)

    for i in range(10):
        if dataset == 'c10':
            idx_class_i = np.where(np.array(clean_train_data.targets) == i)[0]
        elif dataset == 'svhn':
            idx_class_i = np.where(np.array(clean_train_data.labels) == i)[0]
        img_class_i = clean_train_data.data[idx_class_i] / 255

        score_class_i = np.zeros((H * W, 2**C), dtype=float)

        print("searching class {}".format(i))
        for point in tqdm(range(len(score_class_i))):
            point_x = point // H
            point_y = point % H
            for pixel_value in range(2**C):
                channel_value = np.zeros(3)
                channel_value[0] = pixel_value // 2 // 2
                channel_value[1] = pixel_value // 2 % 2
                channel_value[2] = pixel_value % 2
                """objective of searching"""
                if [point, pixel_value] in pts:
                    score_class_i[point, pixel_value] = 0
                else:
                    score_class_i[point, pixel_value] = np.mean(
                        np.abs(channel_value -
                               img_class_i[:, point_x, point_y, :])
                    ) / np.std(
                        np.abs(channel_value -
                               img_class_i[:, point_x, point_y, :])
                    )

        score_class_i_ranking = np.unravel_index(
            np.argsort(score_class_i, axis=None), score_class_i.shape
        )

        for i in range(sparsity):
            max_point, max_pixel_value = (
                score_class_i_ranking[0][-i - 1],
                score_class_i_ranking[1][-i - 1],
            )
            max_point_x, max_point_y = max_point // H, max_point % H
            max_channel_value = np.zeros(3)
            max_channel_value[0] = max_pixel_value // 2 // 2
            max_channel_value[1] = max_pixel_value // 2 % 2
            max_channel_value[2] = max_pixel_value % 2

            pts.append([max_point, max_pixel_value])
            perturbation[idx_class_i, max_point_x, max_point_y, :] = (
                max_channel_value - img_class_i[:, max_point_x, max_point_y, :]
            )

    return perturbation


class Perturbed_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, perturbation, target, transform, pert=1) -> None:
        super().__init__()
        self.data = data
        self.perturbation = perturbation
        self.target = target
        self.transform = transform
        self.pert = pert
        if len(self.perturbation.shape) == 4:
            if self.perturbation.shape[0] == len(self.target):
                self.mode = "S"
            else:
                self.mode = "C"
        else:
            self.mode = "U"

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index: int):
        if self.pert == 1:
            if self.mode == "S":
                img_p, target = (
                    self.data[index] + self.perturbation[index],
                    self.target[index],
                )
            elif self.mode == "C":
                img_p, target = (
                    self.data[index] + self.perturbation[self.target[index]],
                    self.target[index],
                )
            else:
                img_p, target = self.data[index] + \
                    self.perturbation, self.target[index]
        elif self.pert == 2:
            img_p, target = self.perturbation[index], self.target[index]

        else:
            img_p, target = self.data[index], self.target[index]

        img_p = np.clip(img_p, 0, 1)
        img_p = np.uint8(img_p * 255)
        img_p = PIL.Image.fromarray(img_p)
        if self.transform is not None:
            img_p = self.transform(img_p)

        return img_p, target