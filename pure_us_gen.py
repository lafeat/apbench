import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import torch
import numpy as np
from defense.diffusion import diffpure
from defense.ueraser import UEraser
from tqdm import tqdm
import PIL
import os
import cv2
import PIL.Image as Image
from util_us import DatasetPoisoning
from util import CIFAR10_w_indices, NTGA_Dataset_load


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location="center"):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == "center" or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == "random":
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise ("Invalid patch location")

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1:x2, y1:y2, :] = noise
    return mask


class CIFAR10_Transform_TUE(datasets.CIFAR10):
    def __init__(
        self,
        root="data",
        train=True,
        pre_transform=None,
        transform=None,
        download=True,
        perturb_tensor_filepath=None,
        perturbation_budget=1.0,
        samplewise_perturb: bool = False,
        flag_save_img_group: bool = False,
        perturb_rate: float = 1.0,
        clean_train=False,
        in_tuple=False,
        flag_perturbation_budget=False,
    ):
        super(CIFAR10_Transform_TUE, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

        self.samplewise_perturb = samplewise_perturb
        self.pre_transform = pre_transform
        self.in_tuple = in_tuple

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            if flag_perturbation_budget:
                self.noise_255 = (
                    self.perturb_tensor.mul(255 * perturbation_budget)
                    .clamp_(-255 * perturbation_budget, 255 * perturbation_budget)
                    .permute(0, 2, 3, 1)
                    .to("cpu")
                    .numpy()
                )
            else:
                self.noise_255 = (
                    self.perturb_tensor.mul(255 * perturbation_budget)
                    .clamp_(-9, 9)
                    .permute(0, 2, 3, 1)
                    .to("cpu")
                    .numpy()
                )
        else:
            self.perturb_tensor = None
            return

        self.perturbation_budget = perturbation_budget

        if not clean_train:
            if not flag_save_img_group:
                perturb_rate_index = np.random.choice(
                    len(self.targets),
                    int(len(self.targets) * perturb_rate),
                    replace=False,
                )
                self.data = self.data.astype(np.float32)
                for idx in range(len(self.data)):
                    if idx not in perturb_rate_index:
                        continue
                    if not samplewise_perturb:
                        # raise('class_wise still under development')
                        noise = self.noise_255[self.targets[idx]]
                    else:
                        noise = self.noise_255[idx]
                        # print("check it goes samplewise.")
                    noise = patch_noise_extend_to_img(
                        noise, [32, 32, 3], patch_location="center"
                    )
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(
                        self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print("Load perturb done.")
        else:
            print("it is clean train")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, index


class CIFAR10_Transform(datasets.CIFAR10):
    def __init__(
        self, root="data", train=True, pre_transform=None, transform=None, download=True
    ):
        super(CIFAR10_Transform, self).__init__(
            root=root, train=train, download=download, transform=transform
        )
        self.pre_transform = pre_transform
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        for t in self.transform:
            if isinstance(t, DatasetPoisoning):
                img = t(img, target, index)
            else:
                img = t(img)
        return img, target, index


def set_loader(args):
    # construct data loader
    if args.type == "tue":
        train_transform = transforms.ToTensor()
        train_dataset = CIFAR10_Transform_TUE(
            root="dataset/cifar-10/",
            train=True,
            pre_transform=None,
            transform=train_transform,
            download=True,
            perturb_tensor_filepath=f"US_TUE/c10/TUE_{args.arch}.pt",
            perturbation_budget=1.0,
            samplewise_perturb=True,
            clean_train=False,
        )

    elif args.type == "ucl":
        perturb_tensor_filepath = f"US_UCL/c10/UCL_{args.arch}.pt"
        state = torch.load(perturb_tensor_filepath)
        delta = state

        train_transform = [
            transforms.ToTensor(),
            DatasetPoisoning(delta_weight=8 / 255,
                             delta=delta.to("cpu"), args=args),
        ]

        train_dataset = CIFAR10_Transform_UCL(
            root="dataset/cifar-10/",
            train=True,
            pre_transform=None,
            transform=train_transform,
            download=True,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, num_workers=4, pin_memory=True, drop_last=False
    )
    return train_loader


def export_poison(args, advinputs, trainset):
    directory = f"dataset/{args.type}_{args.defense}/"
    path = os.path.join(directory, args.arch)
    if not os.path.exists(path):
        os.makedirs(path)

    def _torch_to_PIL(image_tensor):
        image_denormalized = image_tensor
        image_torch_uint8 = (
            image_denormalized.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
        )
        image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
        return image_PIL

    def _save_image(input, label, idx, location, train=True):
        filename = os.path.join(location, str(idx) + ".png")
        adv_input = advinputs[idx]
        _torch_to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, idx in tqdm(trainset, desc="Dataset generation"):
        _save_image(
            input,
            label,
            idx,
            location=os.path.join(path, "data"),
            train=True,
        )
    print("Dataset fully exported.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c10, c100")
    parser.add_argument("--arch", type=str,
                        default="simclr", help="simclr, moco")
    parser.add_argument("--defense", type=str,
                        default="pure", help="pure, ueraser")
    parser.add_argument(
        "--type",
        default="ucl",
        type=str,
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = set_loader(args)
    adv_inputs, adv_targets = [], []

    for batch_idx, (inputs, targets, index) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.defense == "pure":
            inputs = diffpure(inputs)
        elif args.defense == "ueraser":
            inputs = UEraser(inputs)
        else:
            raise ValueError(args.defense)
        adv_inputs.append(inputs.cpu())
        adv_targets.append(targets.cpu())

    adv_inputs = torch.cat(adv_inputs, dim=0)
    adv_targets = torch.cat(adv_targets, dim=0)
    export_poison(args, adv_inputs, train_loader.dataset)


if __name__ == "__main__":
    main()
