"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict

import torch
import torch.nn.functional as F
import random
import numpy as np
import pdb

from .consts import NON_BLOCKING


def system_startup(args=None, defs=None):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    return setup


def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(
                    np.mean([stat_dict[stat][i] for stat_dict in running_stats])
                )
        else:
            average_stats[stat] = np.mean(
                [stat_dict[stat] for stat_dict in running_stats]
            )
    return average_stats


def cw_loss(outputs, intended_classes, clamp=-100):
    """Carlini-Wagner loss for brewing [Liam's version]."""
    top_logits, _ = torch.max(outputs, 1)
    intended_logits = torch.stack(
        [outputs[i, intended_classes[i]] for i in range(outputs.shape[0])]
    )
    difference = torch.clamp(top_logits - intended_logits, min=clamp)
    return torch.mean(difference)


def reverse_xent(outputs, intended_classes, average=False):
    probs = F.softmax(outputs, dim=1)[0, intended_classes]
    if average:
        return torch.mean(-torch.log(torch.ones_like(probs) - probs))
    else:
        return -torch.log(torch.ones_like(probs) - probs)


"""
def reverse_xent_avg(outputs, intended_classes):
    probs = F.softmax(outputs, dim=1)[torch.arange(len(intended_classes)), intended_classes]
    #probs = F.softmax(outputs, dim=1)[0, intended_classes]
    #return -F.cross_entropy(outputs, intended_classes)
    #return torch.mean(-torch.log(torch.ones_like(probs) - probs))
"""


def reverse_xent_avg(outputs, intended_classes):
    max_exp = outputs.max(dim=1, keepdim=True)[0]
    denominator = torch.log(torch.exp(outputs - max_exp).sum(dim=1)) + max_exp
    other_class_map = torch.tensor(
        [
            [i for i in range(outputs.shape[1]) if i != j]
            for j in range(outputs.shape[1])
        ],
        device=intended_classes.device,
    )
    """
    other_class_map = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 3, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 4, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 5, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 6, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 7, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 8, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 9],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8]], device=intended_classes.device)
    """
    selected_indices = other_class_map[intended_classes]
    other_outputs = outputs.gather(dim=1, index=selected_indices)
    other_max_exp = other_outputs.max(dim=1, keepdim=True)[0]
    numerator = (
        -torch.log(torch.exp(other_outputs - other_max_exp).sum(dim=1)) - other_max_exp
    )
    return torch.mean(numerator + denominator)


def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cw_loss2(outputs, intended_classes, confidence=0, clamp=-100):
    """CW variant 2. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(intended_classes, num_classes=outputs.shape[1])
    target_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - target_logit + confidence, min=clamp)
    return cw_indiv.mean()


def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table_{name}.csv")
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = [line for line in reader][0]
    except Exception as e:
        with open(fname, "w") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        with open(fname, "a") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writerow(kwargs)


def record_results(
    kettle, brewed_loss, results, args, defs, modelkey, extra_stats=dict()
):
    class_names = kettle.trainset.classes
    stats_clean, stats_rerun, stats_results = results

    def _maybe(stats, param, mean=False):
        if stats is not None:
            if len(stats[param]) > 0:
                if mean:
                    return np.mean(stats[param])
                else:
                    return stats[param][-1]

        return ""


def set_random_seed(seed=233):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
