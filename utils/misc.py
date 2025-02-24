"""
File: misc.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Miscellaneous short methods
"""
import torch


def get_num_parameters(model):
    """
    Calculate number of parameters of a model
    :param model: model
    :return: number of parameters in model
    """
    sum_ = 0
    for param in model.parameters():
        if param.requires_grad:  # only sum up values that require grad
            if param.dtype == torch.complex64:  # if type is complex-valued: multiply number of parameters by 2
                # complex parameter equivalent to 2 real-valued parameters
                sum_ += param.numel() * 2
            else:
                sum_ += param.numel()
    return sum_


def mean(l):
    """
    Calculate the mean of a list of numbers
    :param l: list
    :return: mean of list
    """
    return sum(l) / len(l)