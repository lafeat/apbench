import torch
from tqdm import tqdm
import numpy as np
import argparse
import PIL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from attack.basic.em_utils import ResNet_Model
from attack.basic.em_utils import PerturbationTool
import random
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def generate_noises(model, train_loader, noise, noise_generator, optimizer, criterion):
    condition = True
    while condition:
        # optimize theta for M steps
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for j in range(0, 10):
            try:
                (images, labels, index) = next(data_iter)
            except:
                train_idx = 0
                data_iter = iter(train_loader)
                (images, labels, index) = next(data_iter)

            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[train_idx]
                train_idx += 1
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Perturbation over entire dataset
        idx = 0
        for param in model.parameters():
            param.requires_grad = False
        for i, (images, labels, index) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            batch_start_idx, batch_noise = idx, []
            for i, _ in enumerate(images):
                # Update noise to images
                batch_noise.append(noise[idx])
                idx += 1
            batch_noise = torch.stack(batch_noise).cuda()

            # Update sample-wise perturbation
            model.eval()
            images, labels = images.cuda(), labels.cuda()
            perturb_img, eta = noise_generator.min_min_attack(
                images, labels, model, optimizer, criterion, random_noise=batch_noise
            )
            for i, delta in enumerate(eta):
                noise[batch_start_idx + i] = delta.clone().detach().cpu()

        eval_idx, total, correct = 0, 0, 0
        adv_inputs, targets = [], []
        for i, (images, labels, index) in enumerate(train_loader):
            for i, _ in enumerate(images):
                # Update noise to images
                images[i] += noise[eval_idx]
                eval_idx += 1
            images, labels = images.cuda(), labels.cuda()
            adv_inputs.append(images.cpu())
            targets.append(labels.cpu())
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print("Accuracy %.2f" % (acc * 100))
        if acc > 0.99:
            adv_inputs = torch.cat(adv_inputs, dim=0)
            targets = torch.cat(targets, dim=0)
            condition = False
    return adv_inputs, targets



