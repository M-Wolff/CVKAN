"""
File: CVKANWrapper.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Wrapper for CVKAN to make it compatible with our KanPlotter and KanExplainer.
"""
import numpy as np
import torch

from ..CVKAN import CVKAN
from ..functions.ComplexSilu import complex_silu_realweights, complex_silu_complexweight
from .WrapperTemplate import WrapperTemplate


class CVKANWrapper(CVKAN, WrapperTemplate):
    def __init__(self, **kwargs):
        # forward all arguments to superclass (CVKAN)
        super().__init__(**kwargs)
        self.edge_activations = dict()
        self.node_activations = dict()
    def plot_curve(self, l_i_j_index: tuple, num_pts=100) -> (np.ndarray, tuple[np.ndarray]):
        with torch.no_grad():
            l, i, j = list(l_i_j_index)
            # get grid ranges
            grid_min, grid_max = self.get_gridrange(l)
            # construct mesh grid to cover the whole grid evenly
            reals, imags = np.meshgrid(np.linspace(grid_min, grid_max, num_pts),
                                       np.linspace(grid_min, grid_max, num_pts))
            # turn meshgrid into complex numbers
            inputs = torch.complex(torch.from_numpy(reals), torch.from_numpy(imags)).to(torch.complex64)
            # move them to correct device
            inputs = inputs.flatten().to(self.get_kan_layers()[0].realweights.device)

            # grid has shape num_grids x num_grids
            # transform x to shape BATCH x Input-Dim x num_grids x num_grids
            x = inputs.unsqueeze(-1).unsqueeze(-1).expand(inputs.shape + (self.get_kan_layers()[l].num_grids, self.get_kan_layers()[l].num_grids))
            # apply RBF on x (centered around each grid point)
            result = torch.exp(-(torch.abs(x - self.get_kan_layers()[l].grid)) ** 2 / self.get_kan_layers()[l].rho)
            # u and v are the two grid dimensions
            result_real = torch.einsum("buv,uv->b", result, self.get_kan_layers()[l].realweights[i, j, :, :])
            result_imag = torch.einsum("buv,uv->b", result, self.get_kan_layers()[l].complexweights[i, j, :, :])
            # construct complex number out of real and imaginary parts again
            outputs = torch.complex(result_real, result_imag)
            # kind of CSiLU used
            if self.csilu_type == "complex_weight":
                outputs = outputs + self.get_kan_layers()[l].silu_weight[i,j] * complex_silu_complexweight(inputs)
            elif self.csilu_type == "real_weights":
                outputs = outputs + torch.complex(self.get_kan_layers()[l].silu_weight[i, j, 0] * complex_silu_realweights(inputs)[0],
                                                  self.get_kan_layers()[l].silu_weight[i, j, 1] * complex_silu_realweights(inputs)[1])
            # Add SiLU Bias
            outputs +=  self.get_kan_layers()[l].silu_bias[i,j]
            outputs = torch.reshape(outputs, (num_pts, num_pts))
            # TODO include basis and RBFs separately (who cares? maybe for visualization purposes...)
            basis = None
            rbfs = None
        return (reals, imags), (basis, rbfs, outputs.detach().cpu().numpy())

    def get_edge_activations(self, l_i_j):
        l, i, j = l_i_j
        return self.edge_activations[l][:, i, j]

    def get_node_activations(self, l_i):
        l, i = l_i
        return self.node_activations[l][:, i]

    def get_layersizes(self) -> list:
        return self.layers_hidden

    def get_gridrange(self, layer_index) -> list:
        layer = self.layers[layer_index]
        return [layer.grid_min, layer.grid_max]

    def get_kan_layers(self) -> torch.nn.ModuleList:
        return self.layers

    def gather_activations(self, x):
        for layer_idx, layer in enumerate(self.get_kan_layers()):
            # current layer's input are Node activations
            self.node_activations[layer_idx] = x.detach().cpu()
            # x has shape BATCH x Input-Dim
            assert len(x.shape) == 2 and x.shape[1] == layer.input_dim
            # grid has shape num_grids x num_grids
            # transform x to shape BATCH x Input-Dim x num_grids x num_grids
            x_expanded = x.unsqueeze(-1).unsqueeze(-1).expand(x.shape + (layer.num_grids, layer.num_grids,))
            # apply RBF on x_expanded (centered around each grid point)
            result = torch.exp(-(torch.abs(x_expanded - layer.grid)) ** 2 / layer.rho)
            # u and v are the two grid dimensions
            # i and o are input and output indices within layer layer_idx and layer_ix+1
            result_real = torch.einsum("biuv,iouv->bio", result, layer.realweights)
            result_imag = torch.einsum("biuv,iouv->bio", result, layer.complexweights)
            # construct complex number out of real and imaginary parts again
            result_complex = torch.complex(result_real, result_imag)
            assert result_complex.shape[2] == layer.output_dim

            # kind of CSiLU used
            if layer.csilu_type == "complex_weight":
                silu_value = torch.einsum("io,bi->bio", layer.silu_weight, complex_silu_complexweight(x))
            elif layer.csilu_type == "real_weights":
                silu_value_raw = complex_silu_realweights(x)
                silu_value = torch.complex(torch.einsum("io,bi->bio", layer.silu_weight[:, :, 0], silu_value_raw[0]),
                                           torch.einsum("io,bi->bio", layer.silu_weight[:, :, 1], silu_value_raw[1]))
            # Add complex CSiLU bias
            silu_value += layer.silu_bias
            result_complex += silu_value
            # store Edge activations
            self.edge_activations[layer_idx] = result_complex.detach().cpu()
            # sum over all incoming Edges into each Node
            result_complex = torch.einsum("bio->bo", result_complex)
            # potentially apply Normalization
            if layer.use_norm is not None:
                result_complex = layer.norm(result_complex)

            x = result_complex
        self.node_activations[len(self.layers)] = x  # output activations