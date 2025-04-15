import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import kornia.augmentation as K
import random
import torch
import einops
import io
import numpy as np
from PIL import Image

def random_JPEG_compression(in_tensor, quality=None):
    quality = quality or random.randrange(10, 30)
    tensor = in_tensor.mul(255).add_(0.5).clamp_(0, 255)
    tensor = einops.rearrange(tensor, "b c h w -> (b h) w c")
    image = Image.fromarray(tensor.to("cpu", torch.uint8).numpy())
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=quality, optimice=True)
    stream.seek(0)
    tensor = torch.from_numpy(np.asarray(Image.open(stream))).float()
    tensor = tensor.sub_(0.5).div(255)
    tensor = einops.rearrange(tensor, "(b h) w c -> b c h w", b=in_tensor.size(0)).to(
        in_tensor.device
    )
    return (in_tensor - in_tensor) + tensor.detach()

def UEraser(input):
    aug = K.AugmentationSequential(
        K.RandomPlasmaBrightness(
            roughness=(0.3, 0.7),
            intensity=(0.5, 1.0),
            same_on_batch=False,
            p=0.5,
            keepdim=True,
        ),
        K.RandomPlasmaContrast(roughness=(0.3, 0.7), p=0.5),
        K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
        K.auto.TrivialAugment(),
    )
    output = aug(input)
    return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(BasicBlock, self).__init__()
        planes = planes * wide
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(
            mid_planes, self.expansion * planes, kernel_size=1, bias=False
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_dims, out_dims, wide=1):
        super(ResNet, self).__init__()
        self.wide = wide
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            in_dims, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, out_dims)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def resnet18(in_dims, out_dims):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_dims, out_dims, 1)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class PerturbationTool:
    def __init__(
        self,
        seed=0,
        epsilon=0.03137254901,
        num_steps=20,
        step_size=0.00784313725,
        dataset="c10",
    ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.dataset = dataset
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self):
        if self.dataset == "c10" or "svhn":
            noise_shape = [10, 3, 32, 32]
        elif self.dataset == "c100":
            noise_shape = [100, 3, 32, 32]
        elif self.dataset == "imagenet":
            noise_shape = [100, 3, 256, 356]
        else:
            print("Error: Unexpected dataset")
        random_noise = (
            torch.FloatTensor(*noise_shape)
            .uniform_(-self.epsilon, self.epsilon)
            .to(device)
        )

        return random_noise

    def min_min_attack(
        self,
        images,
        labels,
        model,
        optimizer,
        criterion,
        random_noise=None,
    ):
        if random_noise is None:
            random_noise = (
                torch.FloatTensor(*images.shape)
                .uniform_(-self.epsilon, self.epsilon)
                .to(device)
            )

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise

        # Adaptive
        # perturb_img = Variable(UEraser(perturb_img), requires_grad=True)
        
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, "classify"):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(
                perturb_img.data - images.data, -self.epsilon, self.epsilon
            )
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def min_max_attack(
        self,
        images,
        labels,
        model,
        optimizer,
        criterion,
        random_noise=None,
    ):
        if random_noise is None:
            random_noise = (
                torch.FloatTensor(*images.shape)
                .uniform_(-self.epsilon, self.epsilon)
                .to(device)
            )

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(
                perturb_img.data - images.data, -self.epsilon, self.epsilon
            )
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(
        self, noise, image_size=[3, 32, 32], patch_location="center"
    ):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

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
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1:x2, y1:y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
