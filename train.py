import torch.optim as optim
import kornia.augmentation as K
from defense.ueraser import UEraser, UEraser_jpeg
import os
import argparse
import numpy as np
import torch.nn.functional as F
from mardrys import MadrysLoss
from nets import *
import torchvision.models as models
from util import *


def train(model, trainloader, optimizer, criterion, device, epoch, args):
    print("Epoch: %d" % epoch)
    model = torch.nn.DataParallel(model)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    acc = 0
    if args.cutmix:
        cutmix = K.RandomCutMixV2(data_keys=["input", "class"])
    elif args.mixup:
        mixup = K.RandomMixUpV2(data_keys=["input", "class"])

    for batch_idx, (inputs, targets, inputs_clean) in enumerate(trainloader):
        bs = inputs.shape[0]
        num_poisons = bs * args.ratio // 100
        inputs[num_poisons:] = inputs_clean[num_poisons:]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if args.cutmix or args.mixup:
            if args.cutmix:
                inputs, targets = cutmix(inputs, targets)
                targets = targets.squeeze(0)
            else:
                inputs, targets = mixup(inputs, targets)
            outputs = model(inputs)
            loss = loss_mix(targets, outputs)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
            acc += torch.sum(acc_mix(targets, outputs))
            progress_bar(batch_idx, len(trainloader))
            continue
        elif args.ueraser:
            if args.type == "tap" or args.type == "ar":
                U = UEraser_jpeg
            else:
                U = UEraser
            result_tensor = torch.empty((5, inputs.shape[0])).to(device)
            if epoch < args.repeat_epoch:
                for i in range(5):
                    images_tmp = U(inputs)
                    output_tmp = model(images_tmp)
                    loss_tmp = F.cross_entropy(
                        output_tmp, targets, reduction="none")
                    result_tensor[i] = loss_tmp
                outputs = output_tmp
                max_values, _ = torch.max(result_tensor, dim=0)
                loss = torch.mean(max_values)
            else:
                inputs = U(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader))
            continue
        elif args.at:
            outputs, loss = MadrysLoss(epsilon=args.at_eps, distance=args.at_type)(
                model, inputs, targets, optimizer
            )
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader))
    if args.cutmix or args.mixup:
        avg_train_acc = acc * 100.0 / total
    else:
        avg_train_acc = correct * 100.0 / total
    print(f"train_acc: {avg_train_acc:.4f}")
    return avg_train_acc


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader))
    avg_test_acc = correct * 100.0 / total
    print(f"test_acc: {avg_test_acc:.4f}")
    return avg_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c10")
    parser.add_argument(
        "--type",
        default="lsp",
        type=str,
        help="ar, dc, em, rem, hypo, tap, lsp, ntga, ops",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--ratio", default=100,
                        type=int, help="poisoned ratio")

    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--cutout", default=False, action="store_true")
    parser.add_argument("--cutmix", default=False, action="store_true")
    parser.add_argument("--mixup", default=False, action="store_true")
    parser.add_argument("--rnoise", default=False, action="store_true")
    parser.add_argument("--pure", default=False, action="store_true")
    parser.add_argument("--jpeg", default=False, action="store_true")
    parser.add_argument("--bdr", default=False, action="store_true")
    parser.add_argument("--gray", default=False, action="store_true")
    parser.add_argument("--gaussian", default=False, action="store_true")
    parser.add_argument("--nodefense", default=False, action="store_true")

    parser.add_argument("--ueraser", default=False, action="store_true")
    parser.add_argument(
        "--repeat_epoch",
        default=300,
        type=int,
        help="0 for -lite / 50 for UEraser / 300 for -max",
    )

    parser.add_argument("--at", default=False, action="store_true")
    parser.add_argument("--at_eps", default=8 / 255,
                        type=float, help="noise budget")
    parser.add_argument(
        "--at_type", default="L_inf", type=str, help="noise type, [L_inf, L_2]"
    )

    parser.add_argument(
        "--arch", default="r18", type=str, help="r18, r50, se18, mv2, de121"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0
    start_epoch = 0

    # Data
    print("==> Preparing data..")
    transform_train = aug_train(
        args.dataset, args.jpeg, args.gray, args.bdr, args.gaussian, args.cutout, args
    )

    train_set, test_set = get_dataset(args, transform_train)
    train_loader, test_loader = get_loader(args, train_set, test_set)

    if args.dataset == "c100":
        num_classes = 100
    else:
        num_classes = 10

    if args.arch == "r18":
        model = ResNet18(num_classes)
    elif args.arch == "r50":
        model = ResNet50(num_classes)
    elif args.arch == "se18":
        model = SENet18(num_classes)
    elif args.arch == "mv2":
        model = MobileNetV2(num_classes)
    elif args.arch == "de121":
        model = DenseNet121(num_classes)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300], gamma=0.5, last_epoch=-1, verbose=False
    )

    directory = "log"
    path = os.path.join(directory, args.dataset)
    dir = os.path.join(path, args.type)
    d_idx = [
        args.cutout,
        args.cutmix,
        args.mixup,
        args.pure,
        args.jpeg,
        args.bdr,
        args.gray,
        args.gaussian,
        args.nodefense,
        args.ueraser,
        args.at,
        args.nodefense,
    ]
    d_name = [
        "cutout",
        "cutmix",
        "mixup",
        "pure",
        "jpeg",
        "bdr",
        "gray",
        "gaussian",
        "nodefense",
        "ueraser",
        "at",
        "nodefense",
    ]
    defense = d_name[d_idx.index(max(d_idx))]
    if not os.path.exists(dir):
        os.makedirs(dir)
    log_dir = os.path.join(
        dir,
        f"defense={defense}-repeat={args.repeat_epoch}-poison_ratio={args.ratio}-arch={args.arch}.pth",
    )

    train_history, eval_history = [], []
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_acc = train(
            model, train_loader, optimizer, criterion, device, epoch, args
        )
        test_acc = test(model, test_loader, criterion, device)
        train_history.append(train_acc)
        eval_history.append(test_acc)
        scheduler.step()

    print(" Saving...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_history": train_history,
        "eval_history": eval_history,
    }
    torch.save(state, log_dir)


if __name__ == "__main__":
    main()
