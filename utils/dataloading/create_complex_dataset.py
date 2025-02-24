"""
File: create_complex_dataset.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Create a complex-valued dataset based on a lambda function. Similar to pyKAN's create_dataset
method, but complex-valued.
"""
import numpy as np
import torch


def create_complex_dataset(f, n_var, ranges, train_num=1000, test_num=1000, device="cpu"):
    """

    :param f: Lambda expression
    :param n_var: number of input variables
    :param ranges: ranges of input variables from which datapoints are randomly sampled. Possibly one seperate
    range for every input variable, or just one range for all variables combined.
    :param train_num: number of data points to be sampled as train split
    :param test_num: number of data points to be sampled as test split
    :param device: device the dataset should be created on
    :return: Dictionary of dataset (like pyKAN's create_dataset)
    """
    # if ranges is just one single tuple
    if len(np.array(ranges).shape) == 1:
        # copy this tuple to every input_variable (i.e. apply single range tuple to every input variable)
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)
    # prepare tensor (zeros) for train and test input
    train_input = torch.zeros(train_num, n_var, dtype=torch.complex64)
    test_input = torch.zeros(test_num, n_var, dtype=torch.complex64)
    for i in range(n_var):  # for every input variable i
        # sample random points within ranges[i,0] and ranges[i,1] (min max per input variable)
        train_input[:, i] = torch.rand(train_num, dtype=torch.complex64) * (ranges[i, 1] - ranges[i, 0]) + (1+1j)*ranges[i, 0]
        test_input[:, i] = torch.rand(test_num, dtype=torch.complex64) * (ranges[i, 1] - ranges[i, 0]) + (1+1j)*ranges[i, 0]

    # calculate corresponding labels / targets based on lambda expression f
    train_label = f(train_input)
    test_label = f(test_input)

    # if train labels have only 1 dimension (i.e. output is single number per datapoint)
    if len(train_label.shape) == 1:
        # then unsqueeze to Nx1 tensor
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)


    # build dataset
    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset
