import os
import torch
from   torchvision import datasets, transforms
import numpy as np
import logging


TRAIN_IMAGE_FILE_NAME='train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILE_NAME='train-labels-idx1-ubyte.gz'
TEST_IMAGE_FILE_NAME='t10k-images-idx3-ubyte.gz'
TEST_LABEL_FILE_NAME='t10k-labels-idx1-ubyte.gz'
N_CLASSES = 10

""""
def load_train_data(folder, max_n_examples=-1):

    train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

    return train_data.data.numpy(), train_data.targets.numpy()


def load_test_data(folder, max_n_examples=-1):

    test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())

    return test_data.data.numpy(), test_data.targets.numpy()


"""""



def load_train_data(folder, max_n_examples=-1):

    train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

    #return train_data.data.numpy(), train_data.targets.numpy()
    return train_data.data.numpy()[0:max_n_examples], train_data.targets.numpy()[0:max_n_examples]


def load_test_data(folder, max_n_examples=-1):

    test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())

    return test_data.data.numpy(), test_data.targets.numpy()
