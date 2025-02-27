"""
File: PyKANWrapper.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Wrapper for pyKAN to make it compatible with our KanPlotter and KanExplainer.
"""
from datetime import datetime
import numpy as np
import torch

from .WrapperTemplate import WrapperTemplate
from kan import KAN as PyKAN


class PyKANWrapper(PyKAN, WrapperTemplate):
    """
    Wrapper class for PyKAN model to work with our Plotting and Explainer Classes

    Parameter-Counts in PyKAN Layer (requires grad):
    in*out					# Weight for Basis-Scale
    in*out					# Weight for Spline-Scale
    in * out * (num_grids + k)		# Spline-Coefficients
    """

    def __init__(self, **kwargs):
        # Rename Keys to match with other KAN implementations
        # change 'layers_hidden' to 'width'
        kwargs["width"] = kwargs["layers_hidden"]
        self.layers_hidden = kwargs["layers_hidden"]
        del kwargs["layers_hidden"]
        # change 'num_grids' to 'grid'
        kwargs["grid"] = kwargs["num_grids"]
        self.num_grids = kwargs["num_grids"]
        del kwargs["num_grids"]

        # extract parameter 'update_grid' if exists and remove it from kwargs (kwargs will be later forwarded to KAN)
        if "update_grid" in kwargs:
            self._update_grid = kwargs["update_grid"]
            del kwargs["update_grid"]
        else:
            self._update_grid = False
        # set use_norm just for logging of experimental results
        self.use_norm = "grid_update" if self._update_grid else "no"

        # replace 'spline_order' with 'k'
        if "spline_order" in kwargs:
            kwargs["k"] = kwargs["spline_order"]
            del kwargs["spline_order"]
        # create random seed
        if "seed" not in kwargs:
            kwargs["seed"] = int(datetime.now().timestamp())
        # forward remaining kwargs to original KAN implementation
        super().__init__(**kwargs, auto_save=False)
        self.layer_widths = kwargs['width']
        # make sure it's a plain KAN without Multiplication Nodes
        for v in self.layer_widths:
            if v[1] != 0:  # v is a tuple (num_add_nodes, num_mult_nodes))
                raise NotImplementedError('PyKANWrapper: Multiplication Nodes in PyKAN are not yet supported')
        # dictionary for storing Edge and Node Activations
        self.edge_activations = dict()
        self.node_activations = dict()
        self.layers_hidden = [w[0] for w in self.layer_widths]
    def plot_curve(self, l_i_j_index: tuple, num_pts=1000) -> (np.ndarray, tuple[np.ndarray]):
        l,i,j = l_i_j_index
        # extract inputs and outputs from Node and Edge Activations
        inputs = self.get_node_activations((l,i)).cpu().detach().numpy()
        outputs = self.get_edge_activations((l,i,j)).cpu().detach().numpy()
        # they are not ordered yet; order them the same way (sorted based on 'inputs')
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]
        # ToDo: fill None, None with basis function and individual splines if required (only visualization)
        return inputs, (None, None, outputs)

    def get_edge_activations(self, l_i_j_index: tuple):
        layer, i, j = l_i_j_index
        return self.spline_postacts[layer][:,j,i]  # yes, j and i are in 'wrong' order, this is correct

    def get_node_activations(self, l_i_index: tuple):
        layer, i = l_i_index
        return self.acts[layer][:,i]


    def get_layersizes(self) -> list:
        return [w[0] for w in self.layer_widths]

    def get_gridrange(self) -> list:
        raise NotImplementedError("Grid Range return in pykan not yet supported")

    def get_kan_layers(self) -> torch.nn.ModuleList:
        return self.act_fun

    def gather_activations(self, x):
        self.forward(x)

    def fit(self, **kwargs):
        # insert 'update_grid' parameter from initialization time
        if "update_grid" not in kwargs:
            kwargs['update_grid'] = self._update_grid
        super().fit(**kwargs)
