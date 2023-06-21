import torch
from PIL import Image
import os
import sys
import time
import random
from io import BytesIO
import PIL
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


class SubsetImageFolder(datasets.ImageFolder):
    def __init__(
        self, root, transform=None, target_transform=None, subset_percentage=0.2
    ):
        super(SubsetImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.subset_percentage = subset_percentage
        self._create_subset()

    def _create_subset(self):
        self.subset_indices = []
        labels = self.targets
        unique_labels = set(labels)

        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            subset_size = max(
                1, int(len(label_indices) * self.subset_percentage))
            subset_indices = random.sample(label_indices, subset_size)
            self.subset_indices.extend(subset_indices)

        random.shuffle(self.subset_indices)

    def __getitem__(self, index):
        subset_index = self.subset_indices[index]
        image, label = super(SubsetImageFolder, self).__getitem__(subset_index)
        return image, label

    def __len__(self):
        return len(self.subset_indices)


class CIFAR100_w_indices(datasets.CIFAR100):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


class CIFAR10_w_indices(datasets.CIFAR10):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


class SVHN_w_indices(datasets.SVHN):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = PIL.Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


class Dataset_load(torch.utils.data.Dataset):
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
            self.transform(Image.open(
                os.path.join(self.root, self.samples[idx]))),
            label,
            true_img,
        )


TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time

term_width = 80


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def JPEGcompression(image, rate=10):
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=rate, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def aug_train(dataset, jpeg, grayscale, bdr, gaussian, cutout, args):
    transform_train = transforms.Compose([])

    if bdr:
        transform_train.transforms.append(
            transforms.RandomPosterize(bits=2, p=1))
    if grayscale:
        transform_train.transforms.append(transforms.Grayscale(3))
    if jpeg:
        transform_train.transforms.append(transforms.Lambda(JPEGcompression))
    if gaussian:
        transform_train.transforms.append(
            transforms.GaussianBlur(3, sigma=0.1))

    if dataset == "imagenet100":
        if args.clean:
            transform_train.transforms.append(
                transforms.RandomResizedCrop(224))
        else:
            transform_train.transforms.append(transforms.Resize((224, 224)))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            )
        )
        transform_train.transforms.append(transforms.ToTensor())

    elif dataset == "c10":
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "c100":
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "svhn":
        transform_train.transforms.append(transforms.ToTensor())
    if cutout:
        transform_train.transforms.append(Cutout(16))
    return transform_train


def get_dataset(args, transform_train):
    transform_test = transforms.Compose([])
    if args.dataset == "imagenet100":
        transform_test.transforms.append(transforms.Resize((256, 256)))
        transform_test.transforms.append(transforms.CenterCrop(224)),
    transform_test.transforms.append(transforms.ToTensor())
    if args.dataset == "c10":
        base_dataset = CIFAR10_w_indices(
            root=os.environ.get("CIFAR_PATH", "dataset/cifar-10/"),
            train=True,
            download=False,
            transform=transform_train,
        )
    elif args.dataset == "c100":
        base_dataset = CIFAR100_w_indices(
            root=os.environ.get("CIFAR_PATH", "dataset/cifar-100/"),
            train=True,
            download=False,
            transform=transform_train,
        )
    elif args.dataset == "svhn":
        base_dataset = SVHN_w_indices(
            root=os.environ.get("CIFAR_PATH", "dataset/SVHN/"),
            split="train",
            download=True,
            transform=transform_train,
        )
    elif args.dataset == "imagenet100":
        base_dataset = SubsetImageFolder(
            root="dataset/imagenet100/train",
            transform=transform_train,
        )
    else:
        raise ValueError("Valid type and dataset.")

    if args.pure:
        poisons_path = os.path.join("dataset", f"{args.type}_pure")
    else:
        poisons_path = os.path.join("dataset", f"{args.type}_poisons")

    dataset_path = os.path.join(poisons_path, args.dataset)
    type_poisons = ["dc", "em", "rem", "ntga",
                    "hypo", "lsp", "ar", "tap", "ops"]

    if args.type in type_poisons:
        if args.dataset == "imagenet100":
            train_dataset = datasets.ImageFolder(
                root=dataset_path,
                transform=transform_train,
            )
        else:
            dataset_path = os.path.join(dataset_path, "data")
            train_dataset = Dataset_load(
                root=dataset_path, baseset=base_dataset)
    else:
        raise ValueError("Valid type poisons")

    if args.clean:
        train_dataset = base_dataset
    if args.dataset == "c10":
        test_dataset = datasets.CIFAR10(
            root="dataset/cifar-10/",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif args.dataset == "c100":
        test_dataset = datasets.CIFAR100(
            root="dataset/cifar-100/",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif args.dataset == "svhn":
        test_dataset = datasets.SVHN(
            root="dataset/SVHN/",
            split="test",
            download=True,
            transform=transform_test,
        )
    elif args.dataset == "imagenet100":
        test_dataset = SubsetImageFolder(
            root="dataset/imagenet100/val",
            transform=transform_test,
        )

    return train_dataset, test_dataset


def get_loader(args, train_dataset, test_dataset):
    if args.dataset == "imagenet100":
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=True, num_workers=4
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4
        )
    return train_loader, test_loader


def loss_mix(y, logits):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction="none")
    loss_b = criterion(logits, y[:, 1].long(), reduction="none")
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()


def acc_mix(y, logits):
    pred = torch.argmax(logits, dim=1).to(y.device)
    return (1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()
