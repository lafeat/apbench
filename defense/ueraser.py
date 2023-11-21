import kornia.augmentation as K
from PIL import Image
import io
import random
import torch
import einops
import numpy as np


def random_JPEG_compression(in_tensor, quality=None):
    quality = quality or random.randrange(10, 100)
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
            p=1.0,
            keepdim=True,
        ),
        K.RandomPlasmaContrast(roughness=(0.3, 0.7), p=1.0),
        K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
        K.auto.TrivialAugment(),
    )
    output = aug(input)
    return output


def UEraser_jpeg(input):
    aug = K.AugmentationSequential(
        K.RandomPlasmaBrightness(
            roughness=(0.3, 0.7),
            intensity=(0.5, 1.0),
            same_on_batch=False,
            p=0.8,
            keepdim=True,
        ),
        K.RandomPlasmaContrast(roughness=(0.3, 0.7), p=0.8),
        K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
        K.auto.TrivialAugment(),
    )
    tmp = random_JPEG_compression(input)
    tmp = torch.clamp(tmp, 0, 1)
    output = aug(tmp)
    return output


def UEraser_img(input):
    aug = K.AugmentationSequential(
        K.RandomPlasmaBrightness(
            roughness=(0.3, 0.7),
            intensity=(0.3, 1.0),
            same_on_batch=False,
            p=1.0,
            keepdim=True,
        ),
        K.RandomPlasmaContrast(roughness=(0.1, 0.7), p=1.0),
        K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
        K.auto.TrivialAugment(),
    )
    tmp = random_JPEG_compression(input)
    tmp = torch.clamp(tmp, 0, 1)
    output = aug(tmp)
    return output
