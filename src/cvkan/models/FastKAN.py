"""
File: FastKAN.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang; original author: Li Ziyao
Description: Modified copy of FastKAN by Li Ziyao
             (see https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py)
             Changes include:
                - Replaced LayerNorm with BatchNorm
                - Extended plot_curve(...) Method to work with our KANPlotter
                - modified FastKAN.__init__() to include BatchNorm and created attribute to store activations in
                - added method FastKAN.gather_activations(x), which saves all node's and edge's activations to
                  calculate relevance scores later on
                - added functions FastKAN.{get_edge_activations(), get_node_activations(),
                  get_kan_layers(), get_layersizes(), plot_curve()} to support our interface WrapperTemplate
"""
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_batchnorm: bool = True,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:  # if BatchNorm should be used
            self.norm = nn.BatchNorm1d(output_dim, affine=True)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:  # SiLU function
            self.base_activation = base_activation
            # with weight and bias per edge
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        # normalize AFTER current layer (so normalize AFTER layer 0,1,2... but not after the last layer)
        if self.use_batchnorm:
            ret = self.norm(ret)
        return ret

    def plot_curve(
            self,
            input_index: int,
            output_index: int,
            num_pts: int = 1000,
            num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng: (input_index + 1) * ng
            ]  # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )  # num_pts, num_grids
        with (torch.no_grad()):
            y = (w * self.rbf(x.to(w.dtype)))
            # functions is tuple (result_base, result_individual_rbfs, result_all_combined)
            functions = (y, None, None)
            y = y.sum(-1)
            if self.use_base_update:
                base_linear_weight = self.base_linear.weight[output_index, input_index]
                base_value = base_linear_weight * self.base_activation(x)
                base_value += self.base_linear.bias[output_index]
                functions = (base_value, functions[0], None)
                y += base_value
            functions = (functions[0], functions[1], y)

        return x, functions


class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            grid_mins: float = -2.,
            grid_maxs: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
            use_batchnorm = True
    ) -> None:
        super().__init__()
        self.layers_hidden = layers_hidden
        # two lambdas to determine if we need batchnorm in current layer
        is_last_layer = lambda i: i >= len(self.layers_hidden) - 2
        # use batchnorm in every layer except the last one if use_batchnorm is True
        use_batchnorm_after_layer = lambda i: not is_last_layer(i) and use_batchnorm
        self.use_norm = use_batchnorm
        self.num_grids = num_grids
        self.layers = nn.ModuleList()
        # made grid_min and grid_max a list to specify grid intervals for each layer seperately
        if type(grid_mins) != list:
            grid_mins = [grid_mins] * (len(layers_hidden)-1)
        if type(grid_maxs) != list:
            grid_maxs = [grid_maxs] * (len(layers_hidden)-1)
        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs
        for i, (in_dim, out_dim) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            new_layer = FastKANLayer(in_dim, out_dim, grid_mins[i], grid_maxs[i],
                                     num_grids, use_base_update=use_base_update,
                                     base_activation=base_activation,
                                     spline_weight_init_scale=spline_weight_init_scale,
                                     use_batchnorm=use_batchnorm_after_layer(i))
            self.layers.append(new_layer)
        # Dict of Tensors, Index l (layer) contains Tensor of [n_samples x Layer-Width]
        self.node_activations = dict()
        # Dict of Tensors, Index l (layer) contains Tensor of [n_samples x Width Layer l x Width layer l+1]
        self.edge_activations = dict()

    def forward(self, x):
        # assertion to make sure we NEVER leave the grid range of the FIRST layer, as this would indicate a problem
        # with missing normalization of the dataset
        assert torch.min(torch.amin(x, dim=0)) >= self.grid_mins[0] and torch.max(torch.amax(x, dim=0)) <= self.grid_maxs[0], "Inputs into FastKAN should be in range [-1,1]"
        for layer in self.layers:
            x = layer(x)
        return x

    def gather_activations(self, x):
        for l_idx, layer in enumerate(self.layers):
            # Node Activations (Dictionary of Tensors with index specifying layer-index)
            self.node_activations[l_idx] = x
            # Forward pass ( x = layer(x) ) but in more detail
            spline_basis = layer.rbf(x)
            # Spline_linear has dimensions [output_dim x  input_dim * num_grids], transform to [output_dim x input_dim x num_grids]
            spline_linear_pernode = layer.spline_linear.weight.view(layer.spline_linear.weight.shape[0],
                                                                    layer.input_dim, layer.rbf.num_grids)
            # multiply spline_basis (shape [N x input_dim x num_grids]) and spline_linear_pernode
            # rbf_edge_activation has shape [N x input_dim x output_dim]
            rbf_edge_activations = torch.einsum("nig,oig->nio", spline_basis, spline_linear_pernode)

            # silu_edge_activations has shape [N x output_dim]  (so just one per node)
            all_edge_activations = rbf_edge_activations
            if layer.use_base_update:
                silu_edge_activations = torch.einsum("oi,ni->nio", layer.base_linear.weight, layer.base_activation(x))
                # Bias missing, but that doesn't matter for std()
                # silu_edge_activations[:,:,] += layer.base_linear.bias
                all_edge_activations += silu_edge_activations

            self.edge_activations[l_idx] = all_edge_activations

            y = torch.einsum("nio->no", all_edge_activations)
            if layer.use_batchnorm:
                y = layer.norm(y)
            x = y  # current output becomes next input
        self.node_activations[len(self.layers)] = x  # output activations

    def get_edge_activations(self, l_i_j):
        """
        Returns a Tensor of Shape [n_samples] containing all activation values of Edge E_{l,i,j}
        :param l_i_j: Edge index
        :return: All Activation values of Edge with index l_i_j
        """
        l, i, j = l_i_j
        return self.edge_activations[l][:, i, j]

    def get_node_activations(self, l_i):
        """
        Returns a Tensor of Shape [n_samples] containing all activation values of Node N_{l,i}
        :param l_i: Node index
        :return: All Activation values of Node with index l_i
        """
        l, i = l_i
        return self.node_activations[l][:, i]

    def get_kan_layers(self):
        """
        Returns all Layers
        :return: Layers
        """
        return self.layers
    def get_layersizes(self):
        """
        Returns List of integers corresponding to the widths of each layer
        :return: list of layer widths
        """
        return self.layers_hidden
    def plot_curve(self, l_i_j_index):
        """
        Forwards the plot_curve call to the right layer (required by KanPlotter)
        :param l_i_j_index: Edge index
        :return:
        """
        l,i,j = l_i_j_index
        return self.layers[l].plot_curve(input_index=i, output_index=j)


class AttentionWithFastKANTransform(nn.Module):

    def __init__(
            self,
            q_dim: int,
            k_dim: int,
            v_dim: int,
            head_dim: int,
            num_heads: int,
            gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim ** -0.5

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            bias: torch.Tensor = None,  # additive attention bias
    ) -> torch.Tensor:

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm  # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)  # *1khc
        att = (wq * wk).sum(-1).softmax(-2)  # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2], 1, v.shape[-2], self.num_heads, -1)  # *1khc
        o = (att[..., None] * wv).sum(-3)  # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)  # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o