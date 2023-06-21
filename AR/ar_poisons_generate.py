import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
import argparse
import numpy as np
from ar_params import RANDOM_3C_AR_PARAMS_RNMR_10, RANDOM_100CLASS_3C_AR_PARAMS_RNMR_3


class ARProcessPerturb3Channel:
    def __init__(self, b=None):
        super().__init__()
        if b is None:
            self.b = torch.randn((3, 3, 3))
            for c in range(3):
                self.b[c][2][2] = 0
                self.b[c] /= torch.sum(self.b[c])
        # check if b is a torch tensor
        elif type(b) == torch.Tensor:
            self.b = b
        else:
            self.b = torch.tensor(b).float()
        self.num_channels = 3

    def generate(
        self, size=(32, 32), eps=8 / 255, crop=0, p=np.inf, gaussian_noise=False
    ):
        start_signal = torch.randn((self.num_channels, size[0], size[1]))
        kernel_size = 3
        rows_to_update = size[0] - kernel_size + 1
        cols_to_update = size[1] - kernel_size + 1
        ar_coeff = self.b.unsqueeze(dim=1)  # (3, 1, 3, 3)

        for i in range(rows_to_update):
            for j in range(cols_to_update):
                val = torch.nn.functional.conv2d(
                    start_signal[:, i: i + kernel_size, j: j + kernel_size],
                    ar_coeff,
                    groups=self.num_channels,
                )

                # update entry
                noise = torch.randn(1) if gaussian_noise else 0
                start_signal[:, i + kernel_size - 1, j + kernel_size - 1] = (
                    val.squeeze() + noise
                )

        start_signal_crop = start_signal[:, crop:, crop:]

        generated_norm = torch.norm(start_signal_crop, p=p, dim=(0, 1, 2))
        scale = (1 / generated_norm) * eps
        start_signal_crop = scale * start_signal_crop

        return start_signal_crop, generated_norm

    def get_filter(self):
        filter = self.b.detach().clone()
        for c in range(3):
            filter[c][2][2] = -1
        return filter.unsqueeze(dim=0)

    def __repr__(self) -> str:
        return f"{self.b.numpy()}"


def response(filter, signal, response_fn=(lambda x: F.relu(x).sum().item())):
    signal = signal.unsqueeze(dim=0)
    conv_output = torch.nn.functional.conv2d(signal, filter)
    response = response_fn(conv_output)
    return response


def perturb_with_ar_process(ar_processes, inputs, targets, size, crop, eps=1.0):
    batch_size = inputs.size(0)
    adv_inputs = []
    for i in range(batch_size):
        ar_process_perturb = ar_processes[targets[i]]
        delta, _ = ar_process_perturb.generate(
            p=2, eps=eps, size=size, crop=crop)
        adv_input = (inputs[i] + delta).clamp(0, 1)
        adv_inputs.append(adv_input)
    adv_inputs = torch.stack(adv_inputs)
    return adv_inputs


def create_ar_processes(dataset):
    if dataset in ["c10", "svhn"]:
        b_list = RANDOM_3C_AR_PARAMS_RNMR_10
        print(f"Using {len(b_list)} AR processes for {dataset}")
    elif dataset in ["c100"]:
        b_list = RANDOM_100CLASS_3C_AR_PARAMS_RNMR_3
        print(f"Using {len(b_list)} AR processes for {dataset}")
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    ar_processes = []
    for b in b_list:
        ar_p = ARProcessPerturb3Channel(b)
        ar_processes.append(ar_p)

    return ar_processes


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
    ar_processes = create_ar_processes(args.dataset)
    # Data loading code
    if args.dataset == "c10":
        train_dataset = CIFAR10_w_indices(
            root=os.environ.get("CIFAR_PATH", "../dataset/cifar-10/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == "c100":
        train_dataset = CIFAR100_w_indices(
            root=os.environ.get("CIFAR_PATH", "../dataset/cifar-100/"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4
    elif args.dataset == "svhn":
        train_dataset = SVHN_w_indices(
            root=os.environ.get("SVHN_PATH", "../dataset/SVHN/"),
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        noise_size, noise_crop = (36, 36), 4

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=False, num_workers=args.workers
    )
    poison_img = []
    for batch_idx, batch in enumerate(train_loader):
        inputs, target, index = batch
        advinputs = perturb_with_ar_process(
            ar_processes,
            inputs,
            target,
            noise_size,
            noise_crop,
            eps=args.epsilon,
        )
        poison_img.append(advinputs.cpu())
    poison_img = torch.cat(poison_img, dim=0)
    # Save poison
    export_poison(args, poison_img, train_loader.dataset)


def export_poison(args, poison_img, trainset):
    directory = "../dataset/ar_poisons/"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="c10",
                        type=str, help="c10, c100, svhn")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers")
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()
    create_poison(args)


if __name__ == "__main__":
    main()
