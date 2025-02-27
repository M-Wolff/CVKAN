"""
File: eval_model.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Methods to evaluate a given model on arbitrary loss functions as well as for plotting a confusion matrix
for Predictions / GT.
"""
import matplotlib
import numpy as np
import sklearn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def eval_model(model, loss_fns, *, train_data, test_data, train_label, test_label, add_softmax_lastlayer):
    """
    Evaluates a given model on given train and test data as well as ground-truth labels using a dictionary of loss functions
    :param model: model to evaluate
    :param loss_fns: dictionary of loss functions to evaluate the model on
    :param train_data: train input samples (no batching)
    :param test_data: test input samples (no batching)
    :param train_label: train ground truth labels (no batching)
    :param test_label: test ground truth labels (no batching)
    :param add_softmax_lastlayer: whether softmax should be applied after the last layer before evaluation of loss functions
    (CE loss named 'cross_entropy' will ignore this value and always receive raw logits)
    :return: Tuple of two dictionaries (train_losses, test_losses) that each have the names of the loss functions
    evaluated as keys and the resulting value of each loss function as value.
    """
    # dictionaries to store train and test losses for all loss functions to evaluate
    train_losses = dict()
    test_losses = dict()
    # create predictions for train and test input data
    train_predictions = model(train_data)
    test_predictions = model(test_data)
    # iterate over all loss functions
    for loss_fn_name, loss_fn in loss_fns.items():
        current_train_label = train_label
        current_test_label = test_label
        current_train_preds = train_predictions
        current_test_preds = test_predictions
        # if loss function is CE, accuracy, F1 or precision and model output is complex
        if loss_fn_name in ["cross_entropy", "accuracy", "f1", "precision"] and train_predictions.dtype == torch.complex64:
            # only use real parts
            current_train_preds = train_predictions.real
            current_test_preds = test_predictions.real
            # if softmax should be applied (and loss function is not CE; CE requires raw logits)
        if add_softmax_lastlayer and loss_fn_name != "cross_entropy":
            # apply softmax
            current_train_preds = torch.nn.functional.softmax(current_train_preds, dim=1)
            current_test_preds = torch.nn.functional.softmax(current_test_preds, dim=1)
        # some loss functions require argmax and not softmax values
        if loss_fn_name in ["accuracy", "f1", "precision"]:
            current_train_label = torch.argmax(current_train_label, dim=1)
            current_test_label = torch.argmax(current_test_label, dim=1)
            current_train_preds = torch.argmax(current_train_preds, dim=1)
            current_test_preds = torch.argmax(current_test_preds, dim=1)
        # insert train and test losses into the dictionaries with key = name of loss function
        train_losses[loss_fn_name] = loss_fn(current_train_preds, current_train_label)
        test_losses[loss_fn_name] = loss_fn(current_test_preds, current_test_label)
    return train_losses, test_losses

def plot_confusion_matrix(pred, gt, labelmapping=None):
    """
    Plot a confusion matrix based on Predictions and GT values
    :param pred: Predictions
    :param gt: Labels
    :param labelmapping: List of class names in order of class ids for axis descriptions
    """
    # set plot parameters & create figure
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 11}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 16))
    # potentially complex predictions are converted to real part only
    if pred.dtype == torch.complex64:
        pred = pred.real
    # if predictions are softmax values, convert to class ids (argmax)
    if len(pred.shape) > 1:
        pred = pred.argmax(axis=1)
    # if labels are One-Hot encoded values, convert to class ids (argmax)
    if len(gt.shape) > 1:
        gt = gt.argmax(axis=1)
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    # build confusion matrix
    cm = confusion_matrix(y_true=gt, y_pred=pred, normalize="true",labels=[i for i in range(len(labelmapping))] if labelmapping is not None else None)
    # round to 2 decimal places
    cm = np.round(cm, decimals=2)
    # display confusion matrix
    cmd = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labelmapping)
    cmd.plot()
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    #plt.savefig('images/cvkan_knot_confusionmatrix.svg', transparent=True, bbox_inches='tight', pad_inches=0.5)
    plt.show()