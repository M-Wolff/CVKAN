"""
File: loss_functions.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Different loss-functions (MSE, MAE and cross entropy)
"""
import numpy as np
import torch


def mse_formula(pred, label):
    """
    Calculates MSE between prediction and label
    :param pred: prediction
    :param label: label
    :return: MSE value
    """
    pred = pred.squeeze()
    label = label.squeeze()
    assert pred.shape == label.shape, f"Expected Shape {label.shape}, got {pred.shape}"
    if type(pred) == np.ndarray:  # do calculation within numpy
        return np.mean((np.abs(pred-label))**2)
    if type(pred) == torch.Tensor:  # do calculation within torch
        return torch.mean((torch.abs(pred-label))**2)


def mae_formula(pred, label):
    """
    Calculates MAE between prediction and label
    :param pred: prediction
    :param label: label
    :return: MAE value
    """
    pred=pred.squeeze()
    label=label.squeeze()
    assert pred.shape == label.shape
    if type(pred) == np.ndarray:  # do calculation within numpy
        return np.mean(np.abs(pred-label))
    elif type(pred) == torch.Tensor:  # do calculation within torch
        return torch.mean(torch.abs(pred-label))


def cross_entropy(pred, label):
    """
    Calculates cross entropy between prediction and label
    :param pred: prediction
    :param label: label
    :return: cross entropy value
    """
    return torch.nn.functional.cross_entropy(pred, label)


class MSE(torch.nn.Module):
    """
    MSE as a torch module
    """
    def __init__(self):
        super().__init__()
    def forward(self, pred, label):
        return torch.mean((torch.abs(pred - label)) ** 2)


class MAE(torch.nn.Module):
    """
    MAE as a torch module
    """
    def __init__(self):
        super().__init__()
    def forward(self, pred, label):
        return torch.mean(torch.abs(pred - label))
