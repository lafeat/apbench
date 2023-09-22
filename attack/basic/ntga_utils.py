import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10_w_indices(datasets.CIFAR10):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array(img)
        return img, target, index


def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = np.reshape(x, (x.shape[0], -1))
    return (x - np.mean(x)) / np.std(x)


def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return np.reshape(x, (x.shape[0], -1)) / 255


def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset():
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    train_dataset = CIFAR10_w_indices(
        root=os.environ.get("CIFAR_PATH", "../dataset/cifar-10/"),
        train=True,
        download=True,
        transform=None,
    )
    return train_dataset


def accuracy(y_pred, y_test):
    """
    This function calculates the accuracy of mean prediction of Gaussian Process
    :param y_pred: np.ndarray. Prediction of Gaussian Process.
    :param y_test: np.ndarray. Ground truth label.
    :return: a float for accuracy.
    """
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_test, axis=-1))
