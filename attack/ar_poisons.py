import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
import argparse
import numpy as np
from attack.basic.ar_utils import RANDOM_3C_AR_PARAMS_RNMR_10, RANDOM_100CLASS_3C_AR_PARAMS_RNMR_3

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


