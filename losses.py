import numpy as np


def negative_log_likelihood(predicted, actual):
    samples = len(actual)
    correct_logprobs = -np.log(predicted[range(samples), actual])
    data_loss = np.sum(correct_logprobs) / samples
    return data_loss


def nll_derivative(predicted, actual):
    num_samples = len(actual)
    ## compute the gradient on predictions
    dscores = predicted
    dscores[range(num_samples), actual] -= 1
    dscores /= num_samples
    return dscores


def cross_entropy(predicted, actual):
    """Given model outputs (logits) and the indexes of the true class label, computes the softmax cross entropy"""
    true_class_logits = predicted[np.arange(len(predicted)), actual]
    cross_entropy = -true_class_logits + np.log(np.sum(np.exp(predicted), axis=-1))
    return np.mean(cross_entropy)


def cross_entropy_derivative(predicted, actual):
    ones_true_class = np.zeros_like(predicted)
    ones_true_class[np.arange(len(predicted)), actual] = 1
    softmax = np.exp(predicted) / np.exp(predicted).sum(axis=-1, keepdims=True)
    return (-ones_true_class + softmax) / predicted.shape[0]
