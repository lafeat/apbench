import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import PIL
import argparse
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import torchvision.models as models
from attack import ar_poisons, dc_poisons, em_poisons, hypo_poisons, lsp_poisons, ops_poisons, rem_poisons, tap_poisons, ntga_poisons

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
            subset_indices = random.sample(label_indices, subset_size) 
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


def create_poison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Data loading code
    if args.dataset == "c10":
        train_dataset = CIFAR10_w_indices(
            root=os.environ.get("CIFAR_PATH", "./dataset/cifar-10/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == "c100":
        train_dataset = CIFAR100_w_indices(
            root=os.environ.get("CIFAR_PATH", "./dataset/cifar-100/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == "svhn":
        train_dataset = SVHN_w_indices(
            root=os.environ.get("SVHN_PATH", "./dataset/SVHN/"),
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == "imagenet100":
        train_dataset = SubsetImageFolder(
            root="../dataset/imagenet100/train",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=args.workers
    )
    num_classes = 10
    if args.dataset == "c100" or args.dataset == "imagenet100":
        num_classes = 100
    poison_img = []

    if args.type == 'ar':
      ar_processes = ar_poisons.create_ar_processes(args.dataset)
      for batch_idx, (inputs, targets, index) in enumerate(train_loader):
          advinputs = ar_poisons.perturb_with_ar_process(
              ar_processes,
              inputs,
              targets,
              noise_size,
              noise_crop,
              eps=args.eps,
          )
          poison_img.append(advinputs.cpu())
      poison_img = torch.cat(poison_img, dim=0)
    elif args.type == 'dc':
        atknet = dc_poisons.UNet(3).to(device)
        dir = "attack/data/dc_pretrained/atk.0.032.best.pth"
        atknet.load_state_dict(torch.load(dir))
        atknet.eval()
        for batch_idx, (inputs, targets, index) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                noise = atknet(inputs) * args.eps
                atkdata = torch.clamp(inputs + noise, 0, 1)
            poison_img.append(atkdata.detach().cpu())
        poison_img = torch.cat(poison_img, dim=0)
    elif args.type == 'em':
        if args.dataset == "c10" or args.dataset == "c100":
            noise = torch.zeros([50000, 3, 32, 32])
        elif args.dataset == "svhn":
            noise = torch.zeros([73257, 3, 32, 32])
        elif args.dataset == "imagenet100":
            noise = torch.zeros([100000, 3, 224, 224])
        model = em_poisons.ResNet_Model(name="resnet18", num_classes=num_classes)
        model = model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9
        )
        noise_generator = em_poisons.PerturbationTool(
            epsilon=args.eps, num_steps=5, step_size=args.eps / 5
        )
        poison_img, targets = em_poisons.generate_noises(
            model, train_loader, noise, noise_generator, optimizer, criterion
        )
    elif args.type == 'hypo':
        model = hypo_poisons.ResNet18()
        print("Loading checkpoint")
        checkpoint = torch.load("./attack/data/hypo_checkpoint.pth")
        model.load_state_dict(checkpoint["model"])
        model = model.cuda()
        model.eval()
        poison_img = hypo_poisons.poison_hypo(args, train_loader, model)
    elif args.type == 'lsp':
        if args.dataset == "svhn":
            advinputs, targets = lsp_poisons.perturb_with_lsp(
                train_dataset.data, train_dataset.labels, args.dataset
            )
        elif args.dataset == "imagenet100":
            data, targets = [], []
            for idx in tqdm(train_dataset, desc="Data progress"):
                data.append(np.array(idx[0]))
                targets.append(np.array(idx[1]))
            data = np.array(data)
            targets = np.array(targets)
            advinputs, targets = lsp_poisons.perturb_with_lsp(data, targets, args.dataset)
        else:
            advinputs, targets = lsp_poisons.perturb_with_lsp(
                train_dataset.data, train_dataset.targets, args.dataset
            )
        if args.dataset == "imagenet100":
            lsp_poisons.export_imagenet_poison(args, advinputs, targets)
        else:
            lsp_poisons.export_poison(args, advinputs, train_dataset)
    elif args.type == 'ntga':
        x_train_adv, x_train = ntga_poisons.ntga_generate(args.eps, args.dataset)
        ntga_poisons.export_poison(args.dataset, x_train_adv, x_train)
    elif args.type == 'ops':
        images = train_dataset.data.astype(np.float32) / 255
        noise = ops_poisons.pixel_search(train_dataset, sparsity=1)
        if args.dataset == "svhn":
            targets = np.array(train_dataset.labels)
        else:
            targets = np.array(train_dataset.targets)
        poison_img = ops_poisons.Perturbed_Dataset(
            images, noise, targets, transforms.ToTensor())
    elif args.type == 'rem':
        model = rem_poisons.resnet18(3, num_classes)
        if args.dataset == "c10" or args.dataset == "c100":
            noise = torch.zeros([50000, 3, 32, 32])
        elif args.dataset == "svhn":
            noise = torch.zeros([73257, 3, 32, 32])
        elif args.dataset == "imagenet100":
            noise = torch.zeros([100000, 3, 224, 224])
            model = models.resnet18(pretrained=False)
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        model = model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9
        )
        noise_generator = rem_poisons.PerturbationTool(
            epsilon=args.eps, num_steps=5, step_size=args.eps / 5
        )
        noise_generator_adv = rem_poisons.PerturbationTool(
            epsilon=args.eps / 2, num_steps=5, step_size=args.eps / 10
        )
        poison_img, _ = rem_poisons.generate_noises(
            model,
            train_loader,
            noise,
            noise_generator,
            noise_generator_adv,
            optimizer,
            criterion,
        )
    elif args.type == 'tap':
        tap_poisons.tap_gen()
    else:
        raise ValueError("Valid type poisons")

    # Save poison
    if args.type != 'lsp' and args.type != 'tap':
      if args.dataset == "imagenet100":
          export_imagenet_poison(args, poison_img, targets)
      else:
          export_poison(args, poison_img, train_loader.dataset)


def export_poison(args, poison_img, trainset):
    directory = f'./dataset/{args.type}_poisons/'
    path = os.path.join(directory, args.dataset)
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
        adv_input = poison_img[idx]
        _torch_to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, idx in trainset:
        _save_image(input, label, idx, location=os.path.join(
            path, "data"), train=True)
    print("Dataset fully exported.")

def export_imagenet_poison(args, advinputs, targets):
    directory = f'./dataset/{args.type}_poisons/'
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
    parser.add_argument("--dataset", default="c10",
                        type=str, help="c10, c100, svhn, imagenet100")
    parser.add_argument(
        "--type",
        default="ar",
        type=str,
        help="ar, dc, em, rem, hypo, tap, lsp, ntga, ops",
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers")
    parser.add_argument("--eps", type=float, default=8 / 255, help='8 / 255 for L_inf and 1.0 for L_2')
    parser.add_argument("--step_size", default=0.8 / 255, type=float)
    args = parser.parse_args()
    create_poison(args)


if __name__ == "__main__":
    main()
