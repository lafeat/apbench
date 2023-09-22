import argparse
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
import PIL
import os

def comput_l2norm_lim(linf=0.03, feature_dim=3072):
    return np.sqrt(linf**2 * feature_dim)

def normalize_l2norm(data, norm_lim):
    n = data.shape[0]
    orig_shape = data.shape
    flatten_data = data.reshape([n, -1])
    norms = np.linalg.norm(flatten_data, axis=1, keepdims=True)
    flatten_data = flatten_data / norms
    data = flatten_data.reshape(orig_shape)
    data = data * norm_lim
    return data

def perturb_with_lsp(inputs, targets, dataset):
    if dataset == "imagenet100":
        img_size = 224
        noise_frame_size = 32
    else:
        img_size = 32
        noise_frame_size = 8
    n = inputs.shape[0]
    if dataset == "svhn":
        n *= 2
    if dataset == "c100" or dataset == "imagenet100":
        num_classes = 100
    else:
        num_classes = 10
    num_patch = img_size // noise_frame_size
    n_random_fea = int((img_size / noise_frame_size) ** 2 * 3)

    simple_data, simple_label = make_classification(
        n_samples=n,
        n_features=n_random_fea,
        n_classes=num_classes,
        n_informative=n_random_fea,
        n_redundant=0,
        n_repeated=0,
        class_sep=10.0,
        flip_y=0.0,
        n_clusters_per_class=1,
    )
    simple_data = simple_data.reshape(
        [simple_data.shape[0], num_patch, num_patch, 3])
    simple_data = simple_data.astype(np.float32)
    # duplicate each dimension to get 2-D patches
    simple_images = np.repeat(simple_data, noise_frame_size, 2)
    simple_images = np.repeat(simple_images, noise_frame_size, 1)
    simple_data = simple_images[:, 0:img_size, 0:img_size, :]

    # project the synthetic images into a small L2 ball
    linf = 6 / 255.0
    feature_dim = img_size**2 * 3
    l2norm_lim = comput_l2norm_lim(linf, feature_dim)
    simple_data = normalize_l2norm(simple_data, l2norm_lim)

    if dataset == "imagenet100":
        arr_target = np.array(targets)
        for label in range(num_classes):
            orig_data_idx = arr_target == label
            simple_data_idx = simple_label == label
            mini_simple_data = simple_data[simple_data_idx][0: int(
                sum(orig_data_idx))]
            advinputs[orig_data_idx] += mini_simple_data * 255
        advinputs = np.clip((advinputs * 255), 0, 255).astype(np.uint8)
    else:
        advinputs = inputs.astype(np.float32) / 255.0
        if dataset == "svhn":
            advinputs = np.transpose(advinputs, [0, 2, 3, 1])
            arr_target = targets
        else:
            arr_target = np.array(targets)

        # add synthetic noises to original examples
        for label in range(num_classes):
            orig_data_idx = arr_target == label
            simple_data_idx = simple_label == label
            mini_simple_data = simple_data[simple_data_idx][0: int(
                sum(orig_data_idx))]
            advinputs[orig_data_idx] += mini_simple_data

        advinputs = np.clip((advinputs * 255), 0, 255).astype(np.uint8)

    return advinputs, targets

def export_poison(args, advinputs, trainset):
    directory = "../dataset/lsp_poisons/"
    path = os.path.join(directory, args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    def to_PIL(image_ndarray):
        image_PIL = PIL.Image.fromarray(image_ndarray)
        return image_PIL

    def _save_image(input, label, idx, location, train=True):
        filename = os.path.join(location, str(idx) + ".png")
        adv_input = advinputs[idx]
        to_PIL(adv_input).save(filename)

    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    for input, label, idx in tqdm(trainset, desc="Poisoned dataset generation"):
        _save_image(input, label, idx, location=os.path.join(
            path, "data"), train=True)
    print("Dataset fully exported.")


def export_imagenet_poison(args, advinputs, targets):
    directory = "./dataset/lsp_poisons/"
    path = os.path.join(directory, args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    def to_PIL(image_ndarray):
        image_PIL = PIL.Image.fromarray(image_ndarray)
        return image_PIL

    for i in range(100):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
    for j in tqdm(range(advinputs.shape[0])):
        filename = os.path.join(path, str(targets[j]) + "/" + str(j) + ".png")
        to_PIL(advinputs[j]).save(filename)
    print("Dataset fully exported.")

