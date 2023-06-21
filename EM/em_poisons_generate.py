import torch
from tqdm import tqdm
import numpy as np
import argparse
import PIL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import ResNet_Model
from utils import PerturbationTool
import torch.nn as nn
import torchvision.models as models
import random
import os
import cv2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


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
            subset_size = max(1, int(len(label_indices) * self.subset_percentage))
            subset_indices = random.sample(label_indices, subset_size)  # 随机选择子集索引
            self.subset_indices.extend(subset_indices)

        random.shuffle(self.subset_indices)

    def __getitem__(self, index):
        subset_index = self.subset_indices[index]
        image, label = super(SubsetImageFolder, self).__getitem__(subset_index)
        return image, label, subset_index

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


def generate_noises(model, train_loader, noise, noise_generator, optimizer, criterion):
    condition = True
    while condition:
        # optimize theta for M steps
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for j in range(0, 10):
            try:
                (images, labels, index) = next(data_iter)
            except:
                train_idx = 0
                data_iter = iter(train_loader)
                (images, labels, index) = next(data_iter)

            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[train_idx]
                train_idx += 1
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Perturbation over entire dataset
        idx = 0
        for param in model.parameters():
            param.requires_grad = False
        for i, (images, labels, index) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            batch_start_idx, batch_noise = idx, []
            for i, _ in enumerate(images):
                # Update noise to images
                batch_noise.append(noise[idx])
                idx += 1
            batch_noise = torch.stack(batch_noise).cuda()

            # Update sample-wise perturbation
            model.eval()
            images, labels = images.cuda(), labels.cuda()
            perturb_img, eta = noise_generator.min_min_attack(
                images, labels, model, optimizer, criterion, random_noise=batch_noise
            )
            for i, delta in enumerate(eta):
                noise[batch_start_idx + i] = delta.clone().detach().cpu()

        eval_idx, total, correct = 0, 0, 0
        adv_inputs, targets = [], []
        for i, (images, labels, index) in enumerate(train_loader):
            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[eval_idx]
                eval_idx += 1
            images, labels = images.cuda(), labels.cuda()
            adv_inputs.append(images.cpu())
            targets.append(labels.cpu())
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print("Accuracy %.2f" % (acc * 100))
        if acc > 0.99:
            adv_inputs = torch.cat(adv_inputs, dim=0)
            targets = torch.cat(targets, dim=0)
            condition = False
    return adv_inputs, targets


def create_poison(args):
    if args.dataset == "c10":
        train_dataset = CIFAR10_w_indices(
            root=os.environ.get("CIFAR_PATH", "../dataset/cifar-10/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise = torch.zeros([50000, 3, 32, 32])
    elif args.dataset == "c100":
        train_dataset = CIFAR100_w_indices(
            root=os.environ.get("CIFAR_PATH", "../dataset/cifar-100/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise = torch.zeros([50000, 3, 32, 32])
    elif args.dataset == "svhn":
        train_dataset = SVHN_w_indices(
            root=os.environ.get("SVHN_PATH", "../dataset/SVHN/"),
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        noise = torch.zeros([73257, 3, 32, 32])
    elif args.dataset == "imagenet100":
        train_dataset = SubsetImageFolder(
            root="../dataset/imagenet100/train",
            transform=transforms.Compose(
                [transforms.Resize((256, 256)), transforms.ToTensor()]
            ),
        )
        noise = torch.zeros([100000, 3, 224, 224])

    num_classes = 10
    if args.dataset == "c100" or args.dataset == "imagenet100":
        num_classes = 100
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=12,
    )

    model = ResNet_Model(name="resnet18", num_classes=num_classes)
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9
    )
    noise_generator = PerturbationTool(
        epsilon=args.eps, num_steps=5, step_size=args.eps / 3
    )
    adv_inputs, targets = generate_noises(
        model, train_loader, noise, noise_generator, optimizer, criterion
    )
    if args.dataset == "imagenet100":
        export_imagenet_poison(args, adv_inputs, targets)
    else:
        export_poison(args, adv_inputs, train_loader.dataset)


def export_poison(args, advinputs, trainset):
    directory = "../dataset/em_poisons/"
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

    def _save_image(input, label, idx, location, train=True):
        filename = os.path.join(location, str(idx) + ".png")
        adv_input = advinputs[idx]
        _torch_to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, idx in tqdm(trainset, desc="Poisoned dataset generation"):
        _save_image(input, label, idx, location=os.path.join(path, "data"), train=True)
    print("Dataset fully exported.")


def export_imagenet_poison(args, advinputs, targets):
    directory = "../dataset/em_poisons/"
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

    targets = np.array(targets)
    for i in range(100):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
    for j in range(advinputs.shape[0]):
        filename = os.path.join(path, str(targets[j]) + "/" + str(j) + ".png")
        _torch_to_PIL(advinputs[j]).save(filename)
    print("Dataset fully exported.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="c10", type=str, help="c10, c100, svhn, imagenet100"
    )
    parser.add_argument("--eps", default=8 / 255, type=float, help="noise budget")
    args = parser.parse_args()
    create_poison(args)


if __name__ == "__main__":
    main()
