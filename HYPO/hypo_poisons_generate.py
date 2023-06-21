import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import PIL
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os


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


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert not no_relu, "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


class LinfStep(object):
    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        step = torch.sign(g) * self.step_size
        return x - step

    def random_perturb(self, x):
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return new_x


def batch_poison(model, x, target, args):
    orig_x = x.clone().detach()
    step = LinfStep(orig_x, args.eps, args.step_size)

    for _ in range(args.num_steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [x])[0]
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            x = torch.clamp(x, 0, 1)

    return x.clone().detach().requires_grad_(False)


def poison_hypo(args, loader, model):
    poisoned_input = []
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target, index) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        inp_p = batch_poison(model, inp, target, args)
        poisoned_input.append(inp_p.detach().cpu())
    poisoned_input = torch.cat(poisoned_input, dim=0)
    return poisoned_input


def create_poison(args):
    train_dataset = CIFAR10_w_indices(
        root=os.environ.get("CIFAR_PATH", "../dataset/cifar-10/"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    model = ResNet18()
    print("Loading checkpoint")
    checkpoint = torch.load("checkpoint.pth")

    model.load_state_dict(checkpoint["model"])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    advinputs = poison_hypo(args, dataloader, model)

    directory = "../dataset/hypo_poisons/"
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
        adv_input = advinputs[idx]
        _torch_to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, idx in tqdm(
        dataloader.dataset, desc="Poisoned dataset generation"
    ):
        _save_image(input, label, idx, location=os.path.join(path, "data"), train=True)
    print("Dataset fully exported.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="c10", type=str, help="c10")
    parser.add_argument("--eps", default=8 / 255, type=float)
    parser.add_argument("--step_size", default=0.8 / 255, type=float)

    args = parser.parse_args()
    args.num_steps = 100
    create_poison(args)


if __name__ == "__main__":
    main()
