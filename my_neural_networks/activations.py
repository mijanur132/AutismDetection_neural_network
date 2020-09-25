import numpy as np
import torch

EPSILON = 1e-14


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    cross_ent = neg_loglikelihood(X, y_1hot).sum()
    try:
        m = X.shape[1]
    except:
        m = X.data.shape[1]
    return cross_ent / m


def neg_loglikelihood(X, y_1hot, epsilon=EPSILON):
    """Negative Log Likelihood

        Calculate negative log likelihood for each example

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        (n_examples, ).
    """
    X_log = torch.log(X + epsilon)
    nll = -y_1hot * X_log
    return nll.sum(0)


def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    X_max = torch.max(X, 0)[0]
    X_max = X_max.unsqueeze(0).expand_as(X)
    exps = torch.exp(X - X_max)
    exps_sum = torch.sum(exps, 0).unsqueeze(0).expand_as(exps)
    return exps / exps_sum


def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    X_max = torch.max(X, 0)[0]
    X_max = X_max.unsqueeze(0).expand_as(X)
    exps = torch.exp(X - X_max)
    exps_sum = torch.sum(exps, 0).unsqueeze(0).expand_as(exps)
    return exps / exps_sum


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    out = X.clone()
    out[out <= 0] = 0.0
    return out


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)
