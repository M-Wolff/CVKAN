"""
File: crossval_splitter.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Convert dataset with 100% train split to k datasets with non-overlapping test-splits of size (1/k)
using k-fold cross-validation
"""
import torch
from csv_dataloader import CSVDataset

def split_crossval(dataset: CSVDataset, k=5):
    """

    :param dataset: CSVDataset dataset with 100% train split
    :param k: number of folds to do
    :return: list of k datasets, each element representing one fold.
    """
    assert dataset.get_train_test_size()[1] == 0, "For automatic crossvalidation splitting the test set should be empty (100% train)"
    dataset_size = dataset.get_train_test_size()[0]
    crossval_datasets = []
    # loop over all possible split points (0,1,2,...,k-1)
    for split in range(k):
        # calculate test start and end indices
        test_start_index = split * (dataset_size//k)
        test_end_index = (split + 1) * (dataset_size//k)
        # create new dataset dictionary
        data = dict()
        # copy test set from original dataset for current fold based on previously calculated range of indices
        data["test_input"] = dataset.data["train_input"][test_start_index:test_end_index]
        data["test_label"] = dataset.data["train_label"][test_start_index:test_end_index]
        # if test split is surrounded by train data
        if test_start_index > 0 and test_end_index < dataset_size:
            # copy data before and after the test split into the train split
            data["train_input"] = torch.cat((dataset.data["train_input"][0:test_start_index], dataset.data["train_input"][test_end_index:dataset_size, :]))
            data["train_label"] = torch.cat((dataset.data["train_label"][0:test_start_index], dataset.data["train_label"][test_end_index:dataset_size, :]))
        # if test split is the last split
        elif test_start_index > 0 and test_end_index == dataset_size:
            # copy eberything before test split to train split
            data["train_input"] = dataset.data["train_input"][0:test_start_index, :]
            data["train_label"] = dataset.data["train_label"][0:test_start_index, :]
        # if test split is the first split
        elif test_start_index == 0 and test_end_index != dataset_size:
            # copy everything after test split to train split
            data["train_input"] = dataset.data["train_input"][test_end_index:, :]
            data["train_label"] = dataset.data["train_label"][test_end_index:, :]
        # create a CSVDataset object out of the constructed dictionary, copying varnames
        ds = CSVDataset(data, input_vars=dataset.input_varnames, output_vars=dataset.output_varnames, categorical_vars=dataset.categorical_vars)
        # set attribute num_classes the same as in the original datasets, if exists
        if hasattr(dataset, "num_classes"):
            ds.num_classes = dataset.num_classes
        # append current fold's dataset to the list of datasets
        crossval_datasets.append(ds)
    # return list of k datasets
    return crossval_datasets