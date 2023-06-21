import os
import argparse
import time
from PIL import Image
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from util import CIFAR10_w_indices
from util_us import (
    AverageMeter,
    CLmodel,
    SimCLRLoss,
    MoCoLoss,
    adjust_lr,
    linear_eval,
)


class SeparateTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target, index):
        for t in self.transform:
            img = t(img)
        return img


class Dataset_load(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        baseset,
        transform,
        split="train",
        download=False,
    ):
        self.baseset = baseset
        self.transform = transform
        self.samples = os.listdir(root)
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split(".")[0])
        true_img, label, index = self.baseset[true_index]

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [
                sep_transform(Image.open(os.path.join(
                    self.root, self.samples[idx])), label, index),
                sep_transform(Image.open(os.path.join(
                    self.root, self.samples[idx])), label, index),
            ]
        return img, label, index


def set_loader(args):
    # construct data loader
    train_transform = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    base_dataset = CIFAR10_w_indices(
        root=os.environ.get("CIFAR_PATH", "dataset/cifar-10/"),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    if args.type == "tue":
        if args.arch == "simclr":
            train_dataset = Dataset_load(
                root=f"dataset/tue_{args.defense}/simclr/data/",
                baseset=base_dataset,
                transform=train_transform,
            )
        else:
            train_dataset = Dataset_load(
                root=f"dataset/tue_{args.defense}/moco/data/",
                baseset=base_dataset,
                transform=train_transform,
            )
    elif args.type == "ucl":
        if args.arch == "simclr":
            train_dataset = Dataset_load(
                root=f"dataset/ucl_{args.defense}/simclr/data/",
                baseset=base_dataset,
                transform=train_transform,
            )
        else:
            train_dataset = Dataset_load(
                root=f"dataset/ucl_{args.defense}/moco/data/",
                baseset=base_dataset,
                transform=train_transform,
            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, num_workers=4, pin_memory=True, drop_last=True
    )
    return train_loader


def us_train(train_loader, model, criterion, optimizer, epoch, device, args):
    # train clean CL model or re-training CL model on poisoned dataset
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, indexes) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images, labels, indexes = (
            images.to(device),
            labels.to(device),
            indexes.to(device),
        )
        output = model(images, indexes, labels=labels)

        if args.arch == "simclr":
            features = output["features"]
            labels = labels
        elif args.arch == "moco":
            moco_logits = output["moco_logits"]

        bsz = labels.shape[0]

        # compute loss
        if args.arch == "simclr":
            con_loss = criterion(features)
        elif args.arch == "moco":
            con_loss = criterion(moco_logits)

        # update metric
        losses.update(con_loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        con_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print(f"Epoch: {epoch}, Contrastive Loss:{losses.avg}")

    return losses.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="simclr",
                        type=str, help="simclr, moco")
    parser.add_argument("--dataset", default="c10", type=str, help="c10, c100")
    parser.add_argument("--type", default="ucl", type=str, help="tue, ucl")
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--eval_epochs", default=100, type=int)
    parser.add_argument(
        "--defense", default=None, type=str, help="ueraser, pure"
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True
    model = CLmodel(arch=args.arch, dataset=args.dataset, args=args)
    if args.arch == "simclr":
        criterion = SimCLRLoss(temperature=0.5)
    elif args.arch == "moco":
        criterion = MoCoLoss(temperature=0.2)
        args.lr = 0.3

    model = model.to(device)

    optimizer = optim.SGD(
        model.backbone.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    train_loader = set_loader(args)
    start_epoch = 1

    # training routine
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_lr(args, optimizer, epoch)
        time1 = time.time()

        loss = us_train(train_loader, model, criterion,
                        optimizer, epoch, device, args)

        time2 = time.time()

        print(f"epoch:{epoch}, total time:{time2 - time1:.2f}, loss:{loss}")

        # linear probing every eval epochs
        if epoch % args.eval_epochs == 0:
            linear_eval(model, epoch, device, args)

    # save the last model
    directory = "log"
    path = os.path.join(directory, "unsupervised")
    dir = os.path.join(path, args.dataset)
    save_folder = os.path.join(dir, args.type)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = os.path.join(
        save_folder,
        f"type={args.type}-arch={args.arch}-dataset={args.dataset}-defense={args.defense}.pth",
    )

    print("==> Saving...")
    state = {
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(state, save_file)


if __name__ == "__main__":
    main()
