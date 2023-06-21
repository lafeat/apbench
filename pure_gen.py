import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import torch
import numpy as np
from defense.diffusion import diffpure
from tqdm import tqdm
import PIL
import os
import cv2
from util import CIFAR10_w_indices, NTGA_Dataset_load


def get_dataset(args, transform_train):
    if args.dataset == "c10":
        base_dataset = CIFAR10_w_indices(
            root=os.environ.get("CIFAR_PATH", "dataset/cifar-10/"),
            train=True,
            download=False,
            transform=transform_train,
        )
    else:
        raise ValueError("Valid type and dataset.")

    poisons_path = os.path.join("dataset", f"{args.type}_poisons")
    dataset_path = os.path.join(poisons_path, args.dataset)
    type_poisons = ["dc", "em", "rem", "hypo", "lsp", "ar", "tap", "ops", "ntga"]
    if args.type in type_poisons:
        dataset_path = os.path.join(dataset_path, "data")
        train_dataset = dataset_load(root=dataset_path, baseset=base_dataset)
    else:
        raise ValueError("Valid type poisons")
    return train_dataset


class dataset_load(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        baseset,
        split="train",
        download=False,
    ):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(root)
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split(".")[0])
        true_img, label, index = self.baseset[true_index]
        return (
            self.transform(PIL.Image.open(os.path.join(self.root, self.samples[idx]))),
            label,
            true_index,
            idx,
        )


def export_poison(args, advinputs, trainset):
    directory = f"dataset/{args.type}_pure/"
    path = os.path.join(directory, args.dataset)
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

    def _save_image(input, label, true_index, idx, location, train=True):
        filename = os.path.join(location, str(true_index) + ".png")
        adv_input = advinputs[idx]
        _torch_to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, true_index, idx in tqdm(trainset, desc="Pure dataset generation"):
        _save_image(
            input,
            label,
            true_index,
            idx,
            location=os.path.join(path, "data"),
            train=True,
        )
    print("Dataset fully exported.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c10")
    parser.add_argument(
        "--type",
        default="lsp",
        type=str,
        help="ar, dc, em, rem, hypo, tap, lsp, ntga, ops",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_train = transforms.ToTensor()
    train_set = get_dataset(args, transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=False, num_workers=4, drop_last=False
    )

    adv_inputs, adv_targets = [], []

    for batch_idx, (inputs, targets, index, index1) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = diffpure(inputs)
        adv_inputs.append(inputs.cpu())
        adv_targets.append(targets.cpu())

    adv_inputs = torch.cat(adv_inputs, dim=0)
    adv_targets = torch.cat(adv_targets, dim=0)
    export_poison(args, adv_inputs, train_loader.dataset)


if __name__ == "__main__":
    main()
