import torch
from tqdm import tqdm
import numpy as np
import argparse
from attack.basic.rem_utils import resnet18
from attack.basic.rem_utils import PerturbationTool
import PIL
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def generate_noises(
    model,
    train_loader,
    noise,
    noise_generator,
    noise_generator_adv,
    optimizer,
    criterion,
):
    condition = True
    rnoise = noise.clone()
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

        idx, id = 0, 0
        for param in model.parameters():
            param.requires_grad = False
        for i, (images, labels, index) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            # Update sample-wise perturbation
            model.eval()
            images, labels = images.cuda(), labels.cuda()

            # adv funciton
            batch_start_id, batch_rnoise = id, []

            for i, _ in enumerate(images):
                # Update rnoise to images
                batch_rnoise.append(rnoise[id])
                id += 1
            batch_rnoise = torch.stack(batch_rnoise).cuda()

            _, reta = noise_generator_adv.min_max_attack(
                images, labels, model, optimizer, criterion, random_noise=batch_rnoise
            )
            for i, delta in enumerate(reta):
                rnoise[batch_start_id + i] = delta.clone().detach().cpu()
                noise[batch_start_id + i] = delta.clone()

            # em funciton
            batch_start_idx, batch_noise = idx, []
            for i, _ in enumerate(images):
                # Update noise to images
                batch_noise.append(noise[idx])
                idx += 1
            batch_noise = torch.stack(batch_noise).cuda()

            _, eta = noise_generator.min_min_attack(
                images, labels, model, optimizer, criterion, random_noise=batch_noise
            )
            for i, delta in enumerate(eta):
                noise[batch_start_idx + i] = delta.clone().detach().cpu()

        eval_idx, total, correct = 0, 0, 0
        adv_inputs = []
        for i, (images, labels, index) in enumerate(train_loader):
            for i, _ in enumerate(images):
                images[i] += noise[eval_idx]
                eval_idx += 1
            images, labels = images.cuda(), labels.cuda()
            adv_inputs.append(images.cpu())
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print("Accuracy %.2f" % (acc * 100))
        if acc > 0.99:
            adv_inputs = torch.cat(adv_inputs, dim=0)
            condition = False
    return adv_inputs, noise


def poisoned_dataset(args, noise, inputs, targets):
    perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu").numpy()
    inputs = inputs.astype(np.float32)

    if args.dataset == "svhn":
        inputs = np.transpose(inputs, [0, 2, 3, 1])
        arr_target = targets
    else:
        arr_target = np.array(targets)
    for i in range(len(inputs)):
        inputs[i] += perturb_noise[i]

    advinputs = np.clip(inputs, 0, 255).astype(np.uint8)
    return advinputs, arr_target



