"""
File: run_crossval.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Running k-fold cross-validation training runs and logging the results with additional meta-data
             in a json file
"""
import copy
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from ..train.train_loop import train_kans
from ..utils.dataloading.crossval_splitter import split_crossval
from ..utils.dataloading.csv_dataloader import CSVDataset
from ..utils.misc import get_num_parameters


def run_crossval(model, dataset_full_train: CSVDataset, dataset_name, loss_fn_backprop, loss_fns, batch_size, device=torch.device("cuda"), logging_interval=100, add_softmax_lastlayer=False, epochs=500, convert_model_output_to_real=True, k=5):
    """
    Runs cross-validaiton on a given dataset
    :param model: Model to train
    :param dataset_full_train: CSVDataset object that only contains training data and zero test data for splitting into k folds
    :param dataset_name: Name of the dataset (for logging purposes only)
    :param loss_fn_backprop: Loss function to use for backpropagation
    :param loss_fns: dictionary of additional loss functions (also including loss_fn_backprop) to evaluate and save
    :param batch_size: batch size to use for training. Should be -1 for pyKAN models!
    :param device: device to train on
    :param logging_interval: how often to display logs in console
    :param add_softmax_lastlayer: flag whether softmax should be added after the last layer
    :param epochs: epochs to train for
    :param convert_model_output_to_real: should be True only if model produces complex-valued output but
    we need real-valued output. Essentially does model(x).real instead of only model(x)
    :param k: number of folds to do cross-validation on
    :return: None
    """
    # generate k-fold cv (usually k=5)
    datasets = split_crossval(dataset_full_train, k=k)  # returns list of datasets (with different crossval splits each)
    num_folds = len(datasets)
    # build dictionary to store the results into
    results = dict()
    results["train_losses"] = dict()
    results["test_losses"] = dict()
    # mean and std across the k folds
    results["train_losses"]["mean"] = dict()
    results["train_losses"]["std"] = dict()
    results["test_losses"]["mean"] = dict()
    results["test_losses"]["std"] = dict()
    results["model_name"] = model.__class__.__name__
    results["dataset_size"] = dataset_full_train.get_train_test_size()
    # make sure loss_fn_backprop is a class and not just a lambda or method.
    results["loss_fn_backprop"] = loss_fn_backprop.__class__.__name__
    results["batch_size"] = batch_size
    results["epochs"] = epochs
    results["complex"] = convert_model_output_to_real
    results["add_softmax_lastlayer"] = add_softmax_lastlayer
    results["layers"] = model.layers_hidden
    results["use_norm"] = model.use_norm
    results["num_grids"] = model.num_grids
    results["rho"] = model.rho if hasattr(model, "rho") else None
    results["dataset_name"] = dataset_name
    results["kfolds"] = k
    results["num_trainable_params"] = get_num_parameters(model)
    results["zsilu_type"] = model.csilu_type if hasattr(model, "csilu_type") else None
    results["start_timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # iterate over the k folds
    for i, d in enumerate(datasets):
        results["train_losses"][i] = dict()
        results["test_losses"][i] = dict()
        my_model = copy.deepcopy(model)  # create a copy of the model
        # run training on current fold
        train_loss, test_loss = train_kans(my_model, d, loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                                           device=device, batch_size=batch_size, logging_interval=logging_interval,
                                           add_softmax_lastlayer=add_softmax_lastlayer, epochs=epochs,
                                           last_layer_output_real=convert_model_output_to_real)
        # and store results for each loss_fn given
        for k in loss_fns.keys():
            results["train_losses"][i][k] = train_loss[k].item()
            results["test_losses"][i][k] = test_loss[k].item()

    # calculate mean and std for all loss_fn's given
    for k in loss_fns.keys():
        train_losses_per_function = np.array([results["train_losses"][i][k] for i in range(num_folds)])
        test_losses_per_function = np.array([results["test_losses"][i][k] for i in range(num_folds)])
        results["train_losses"]["mean"][k] = np.mean(train_losses_per_function)
        results["train_losses"]["std"][k] = np.std(train_losses_per_function)
        results["test_losses"]["mean"][k] = np.mean(test_losses_per_function)
        results["test_losses"]["std"][k] = np.std(test_losses_per_function)
    results["finish_timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(results)
    results_file = Path(os.path.abspath(__file__)).parent / "results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(results)
    with open(results_file, "w+", encoding="utf-8") as f:
        json.dump(all_results, f)
