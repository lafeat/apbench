import torch
import math
from PIL import Image
import os
import time
from torchvision import transforms, datasets
import numpy as np
import torch.nn.functional as F
from defense.ueraser import UEraser
from defense.diffusion import diffpure
import torch.nn as nn
from nets.resnet_us import model_dict, LinearClassifier
import kornia.augmentation as K


class I_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class I_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class DatasetPoisoning(object):
    def __init__(self, delta_weight, delta, args):
        self.delta_weight = delta_weight
        self.delta = delta
        self.args = args

    def __call__(self, img, target, index):
        img = torch.clamp(
            img + self.delta_weight * torch.clamp(self.delta[index], min=-1.0, max=1.0),
            min=0.0,
            max=1.0,
        )
        return img

    def __repr__(self):
        return "Adding pretrained noise to dataset (using poisoned dataset) when re-training"


class Defense(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target, index):
        img = self.transform(img)
        img = img.squeeze(0)
        return img


def ueraser_transform(imgs):
    imgs = UEraser(imgs)
    return imgs


def pure_transform(imgs):
    imgs = diffpure(imgs)
    return imgs


class SeparateTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target, index):
        for t in self.transform:
            img = t(img)
        return img


class NormalTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target, index):
        for t in self.transform:
            if isinstance(t, DatasetPoisoning) or isinstance(t, Defense):
                img = t(img, target, index)
            else:
                img = t(img)
        return img


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


class TransferCIFAR10Pair(datasets.CIFAR10):
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
        super(TransferCIFAR10Pair, self).__init__(
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
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print("Load perturb done.")
        else:
            print("it is clean train")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.pre_transform is not None:
            nor_transform = NormalTransform(self.pre_transform)
            img = nor_transform(img, target, index)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class TransferCIFAR100Pair(datasets.CIFAR100):
    def __init__(
        self,
        root="data",
        train=True,
        pre_transform=None,
        transform=None,
        download=True,
        perturb_tensor_filepath=None,
        perturbation_budget=1.0,
        samplewise_perturb: bool = True,
        flag_save_img_group: bool = False,
        perturb_rate: float = 1.0,
        clean_train=False,
        in_tuple=False,
        flag_perturbation_budget=False,
        args=None,
    ):
        super(TransferCIFAR100Pair, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

        self.samplewise_perturb = samplewise_perturb
        self.pre_transform = pre_transform
        self.in_tuple = in_tuple
        self.args = args

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
                        noise = self.noise_255[self.targets[idx]]
                    else:
                        noise = self.noise_255[idx]
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print("Load perturb done.")
        else:
            print("it is clean train")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.pre_transform is not None:
            nor_transform = NormalTransform(self.pre_transform)
            img = nor_transform(img, target, index)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class P_CIFAR10_TwoCropTransform(datasets.CIFAR10):
    def __init__(
        self, root="data", train=True, pre_transform=None, transform=None, download=True
    ):
        super(P_CIFAR10_TwoCropTransform, self).__init__(
            root=root, train=train, download=download, transform=transform
        )
        self.pre_transform = pre_transform
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.pre_transform is not None:
            nor_transform = NormalTransform(self.pre_transform)
            img = nor_transform(img, target, index)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class P_CIFAR100_TwoCropTransform(datasets.CIFAR100):
    def __init__(
        self,
        root="data",
        train=True,
        pre_transform=None,
        transform=None,
        download=True,
    ):
        super(P_CIFAR100_TwoCropTransform, self).__init__(
            root=root, train=train, download=download, transform=transform
        )
        self.pre_transform = pre_transform
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.pre_transform is not None:
            nor_transform = NormalTransform(self.pre_transform)
            img = nor_transform(img, target, index)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def set_optimizer(args, model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(1)]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        return grad_out


class ResNetWithHead(nn.Module):
    """backbone + projection head"""

    def __init__(self, arch="resnet18", head="mlp", feat_dim=128):
        super(ResNetWithHead, self).__init__()
        model_fun, dim_in = model_dict[arch]
        self.encoder = model_fun()
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class MoCo(nn.Module):
    def __init__(
        self,
        base_encoder,
        arch="resnet18",
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        allow_mmt_grad=False,
    ):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.allow_mmt_grad = allow_mmt_grad

        self.encoder_q = base_encoder(
            arch=arch, head="mlp" if mlp else "linear", feat_dim=dim
        )
        self.encoder_k = base_encoder(
            arch=arch, head="mlp" if mlp else "linear", feat_dim=dim
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        # compute key features
        with torch.set_grad_enabled(self.allow_mmt_grad):  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k if not self.allow_mmt_grad else k.clone().detach())

        return logits


class CLmodel(nn.Module):
    def __init__(self, arch, dataset, args):
        super(CLmodel, self).__init__()
        self.arch = arch
        self.dataset = dataset
        self.args = args
        if args.arch == "simclr":
            self.backbone = ResNetWithHead(arch="resnet18")
        elif args.arch == "moco":
            self.backbone = MoCo(
                ResNetWithHead,
                arch="resnet18",
                dim=128,
                K=4096,
                m=0.99,
                T=0.2,
                mlp=True,
                allow_mmt_grad=False,
            )
        else:
            raise ValueError(args.arch)

        self.transform = nn.Sequential(
            K.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
        )

    def forward(self, img, index, labels=None):
        mixed_img = img
        bsz = img.shape[0] // 2
        # data augmentation

        aug1, aug2 = torch.split(mixed_img, [bsz, bsz], dim=0)
        aug = torch.cat([aug1, aug2], dim=0)

        out_dict = {}
        if self.args.arch == "simclr":
            features = self.backbone(aug)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            out_dict["features"] = features
        elif self.args.arch == "moco":
            moco_logits = self.backbone(im_q=aug1, im_k=aug2.detach())
            out_dict["moco_logits"] = moco_logits
        else:
            raise ValueError(self.args.arch)

        return out_dict


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MoCoLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, logits, labels=None, queue_labels=None):
        """
        logits: Nx(1+K)
        labels: N,
        queue_labels: K,
        """
        device = torch.device("cuda") if logits.is_cuda else torch.device("cpu")
        # CL loss
        bsz = logits.shape[0]
        if labels is None and queue_labels is None:
            mask = torch.zeros_like(logits)
            mask[:, 0] = 1.0
        else:
            labels = labels.contiguous().view(-1, 1)
            queue_labels = queue_labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, queue_labels.T).float().to(device)  # NxK
            mask = torch.cat([torch.ones(bsz, 1).to(device), mask], dim=1)  # Nx(K+1)

        logits /= self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def adjust_lr(args, optimizer, epoch):
    lr = args.lr

    eta_min = lr * (args.lr_decay_rate**3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, args):
    # training linear classifier
    model.eval()
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg


def validate_linear(val_loader, model, classifier, criterion, args):
    # validating linear classifier
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            # update metric
            losses.update(loss.item(), bsz)
            top1.update(acc1.item(), bsz)

    return losses.avg, top1.avg


def train_val_linear(model, device, args):
    # linear probing
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.ToTensor()
    best_acc = 0
    epochs = 100

    if args.dataset == "c10":
        train_dataset = datasets.CIFAR10(
            root="dataset/cifar10/", transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="dataset/cifar10/", train=False, transform=val_transform
        )
        num_classes = 10
    else:
        train_dataset = datasets.CIFAR100(
            root="dataset/cifar100/", transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR100(
            root="dataset/cifar100/", train=False, transform=val_transform
        )
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, pin_memory=True
    )
    classifier = LinearClassifier(arch="resnet18", num_classes=num_classes)
    linear_criterion = torch.nn.CrossEntropyLoss()
    classifier = classifier.to(device)
    linear_optimizer = set_optimizer(args=args, model=classifier)

    # training routine
    for epoch in range(1, epochs + 1):
        adjust_lr(args, linear_optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train_linear(
            train_loader,
            model,
            classifier,
            linear_criterion,
            linear_optimizer,
            epoch,
            args,
        )

        # eval for one epoch
        val_loss, val_acc = validate_linear(
            val_loader, model, classifier, linear_criterion, args
        )

        print(f"Train epoch:{epoch}, val acc {val_acc}")
        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


def linear_eval(model, epoch, device, args):
    print(f"================== Epoch [{epoch}] =====================")
    if args.arch == "simclr":
        eval_model = model.backbone
    elif args.arch == "moco":
        eval_model = model.backbone.encoder_q

    acc = train_val_linear(eval_model, device, args)
    print(f"Epoch {epoch} | ***best linear_acc {acc:.2f}")
