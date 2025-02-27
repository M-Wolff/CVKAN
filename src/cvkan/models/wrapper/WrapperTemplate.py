"""
File: WrapperTemplate.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Interface for KANs. To make them work with our KanPlotter and KanExplainer all the methods need
             to be supported by the specific KAN variant/wrapper.
"""

from abc import abstractmethod

import numpy as np
import torch


class WrapperTemplate:
    def __init__(self):
        pass
    @abstractmethod
    def plot_curve(self, l_i_j_index: tuple, num_pts = 1000) -> (np.ndarray, tuple[np.ndarray]):
        """
        Returns the learned Activation Function with index (l,i,j) connecting Node i in Layer l to Node j in Layer l+1
        :param l_i_j_index: Tuple for Index (l,i,j)
        :param num_pts: Number of points for sampling
        :return: Tuple (xs, (ys_base, ys_splines, ys_sum_total)) with input Values and output Values per component
        xs are input values, ys_base the base activation (i.e. SiLU), ys_splines the individual Splines/RBFs/... and
        ys_sum_total is the sum of everything (so the final learned function)
        """
        raise NotImplementedError
    @abstractmethod
    def get_edge_activations(self, l_i_j_index: tuple):
        """
        Returns Activations of the Edge with Index (l,i,j)
        :param l_i_j_index: Tuple (l,i,j)
        :return: Activations across all Samples for given Edge provided in 'gather_activations(...)' with Dimension [N]
        """
        raise NotImplementedError
    @abstractmethod
    def get_node_activations(self, l_i_index: tuple):
        """
        Returns Activations of the Node with Index (l,i)
        :param l_i_index: Tuple (l,i)
        :return: Activations across all Samples for given Node provided in 'gather_activations(...)' with Dimension [N]
        """
        raise NotImplementedError
    @abstractmethod
    def get_layersizes(self) -> list:
        """
        Returns the widths of all layers
        :return: List with one value for each layer starting at the bottom (input side) of the Network representing
        layer's width
        """
        raise NotImplementedError
    @abstractmethod
    def get_gridrange(self) -> list:
        """
        Returns the grid-range as a List
        :return: Grid Range
        """
        # ToDo: maybe change this for only one single Activation Function, not for the whole Network/Layer
        raise NotImplementedError
    @abstractmethod
    def get_kan_layers(self) -> torch.nn.ModuleList:
        """
        Returns all Layers as a ModuleList
        :return: ModuleList of all Layers
        """
        raise NotImplementedError
    @abstractmethod
    def gather_activations(self, x):
        """
        Collects the Activations on every Edge and Node for later Usage
        :param x: Input Samples
        """
        raise NotImplementedError
    def get_all_weights(self):
        """
        Returns all weights as a concatenated, flattened Tensor. To be used for Regularization.
        :return: Concatenated, flattened Tensor with all trainable Weights
        """
        raise NotImplementedError