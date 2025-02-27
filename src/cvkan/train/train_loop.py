"""
File: train_loop.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Main loop for training all kinds of KANs on any dataset with arbitrary loss functions
"""
import torch
from torch.utils.data import DataLoader

from ..models.wrapper import PyKANWrapper
from ..utils.dataloading.csv_dataloader import CSVDataset
from ..utils.eval_model import eval_model
from ..utils.misc import get_num_parameters


def train_kans(model, dataset: CSVDataset, loss_fn_backprop, loss_fns, device=torch.device("cuda"), epochs=5000,
               batch_size=1000, kan_explainer=None, logging_interval=50, add_softmax_lastlayer=False, last_layer_output_real=True, sparsify=False):

    """
    Train KANs for different datasets. Suitable for complex and real-valued KANs
    :param model: model to train
    :param dataset: dataset to train on (has to be instance of class CSVDataset)
    :param loss_fn_backprop: loss function to be used for backpropagation
    :param loss_fns: dictionary of loss functions to evaluate the model on (key of dict entry should be loss function name)
    :param device: device to train on (default Cuda)
    :param epochs: number of epochs to train for
    :param batch_size: batch size
    :param kan_explainer: kan_explainer to use for regularization of edge's importance scores. Only used if 'sparsify=True'
    :param logging_interval: interval in which results should be logged on console
    :param add_softmax_lastlayer: flag whether softmax should be applied before evaluation the entries of 'loss_fns' dict
    Keep in mind that loss function named 'cross_entropy' will not have SoftMax applied beforehand in any case.
    :param last_layer_output_real: useful for classification tasks where the last layer needs to be real. Simply takes
    the real part of the model's output
    :return: resulting train and test losses at the very end of training. Both of them are a dict with keys representing the
    name of the loss function (same as keys in parameter 'loss_fns')
    """
    # __getitem__ of dataset should give elements from train split
    dataset.set_returnsplit("train")
    # move model to correct device
    model.to(device)
    if type(model) == PyKANWrapper:  # pyKAN (needs to be wrapper, because pykan does not store certain attributes by itself...)
        # move dataset (without batching) to correct device
        dataset.to(device)
        # normalize dataset for pykan
        dataset.normalize_for_pykan()
        # update the model's grid from train samples
        model.update_grid_from_samples(dataset.data["train_input"])
        # pykan needs an eval-loss function that can be executed **without** any parameters. So the dataset (test split)
        # has to be hard-coded into the loss function for some reason...
        def hardcoded_pykan_testloss_metric():
            """Why would anybody program it like this?! Why not just make test GT and Prediction a parameter
            like every other loss function?..."""
            return loss_fn_backprop(model(dataset.data["test_input"].to(device)), dataset.data["test_label"].to(device))
        # fit the model
        model.fit(dataset=dataset.data, opt="LBFGS", steps=epochs, loss_fn=loss_fn_backprop, batch=batch_size,
                  metrics = [hardcoded_pykan_testloss_metric], display_metrics=["hardcoded_pykan_testloss_metric"])
        # evaluate the trained model on all loss functions
        train_losses, test_losses = eval_model(model, loss_fns, train_data=dataset.data["train_input"],
                                    test_data=dataset.data["test_input"],
                                    train_label=dataset.data["train_label"],
                                    test_label=dataset.data["test_label"],
                                    add_softmax_lastlayer=add_softmax_lastlayer)
        print(
            f"Final Train Loss: {[(lfn, l.item()) for lfn, l in train_losses.items()]}, Final Test Loss: {[(lfn, l.item()) for lfn, l in test_losses.items()]}")
        return train_losses, test_losses
    # if model is not PyKAN Wrapper, check if batch_size is > 0 (for pykan batch size should be -1)
    assert batch_size > 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Number of trainable parameters in the model: ", get_num_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # decay learning rate by 0.6 every epochs//10 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 10, gamma=0.6)

    # train loop
    for epoch in range(epochs):
        # evaluate without grads
        with torch.no_grad():
            if epoch % logging_interval == 0:
                train_losses, test_losses = eval_model(model, loss_fns, train_data=dataset.data["train_input"].to(device),
                                                       test_data=dataset.data["test_input"].to(device),
                                                       train_label=dataset.data["train_label"].to(device),
                                                       test_label=dataset.data["test_label"].to(device),
                                                       add_softmax_lastlayer=add_softmax_lastlayer)
                print(
                    f"Epoch {epoch} Train Loss: {[(lfn, l.item()) for lfn, l in train_losses.items()]}, Test Loss: {[(lfn, l.item()) for lfn, l in test_losses.items()]}, LR: {scheduler.get_last_lr()}")

        # iterate over all batches in train dataloader
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            # potentially convert model's output to real
            if last_layer_output_real:
                train_predictions = model(X).real
            else:
                train_predictions = model(X)
            # calculate loss for backprop
            train_loss = loss_fn_backprop(train_predictions, y)
            if sparsify:  # potentially regularize the net using KANExplainer (Edge's relevance scores)
                sparsity_regularization = 0
                kan_explainer.calc_relevances_pykan()
                for k in kan_explainer.edge_relevances.keys():
                    sparsity_regularization += 1*(kan_explainer.get_edge_relevance(k).abs()).sum()
                train_loss = train_loss + sparsity_regularization
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
    # Final evaluation
    train_losses, test_losses = eval_model(model, loss_fns, train_data=dataset.data["train_input"].to(device),
                                           test_data=dataset.data["test_input"].to(device),
                                           train_label=dataset.data["train_label"].to(device),
                                           test_label=dataset.data["test_label"].to(device),
                                           add_softmax_lastlayer=add_softmax_lastlayer)
    print(
        f"Final Train Loss: {[(lfn, l.item()) for lfn, l in train_losses.items()]}, Final Test Loss: {[(lfn, l.item()) for lfn, l in test_losses.items()]}")
    return train_losses, test_losses