"""
File: CVKAN.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: CVKAN Model definition
"""
from typing import List
import torch


from .functions.CV_LayerNorm import Complex_LayerNorm, Complex_BatchNorm, Complex_BatchNorm_naiv, \
    Complex_BatchNorm_var

from .functions.ComplexSilu import complex_silu_complexweight, complex_silu_realweights

class Norms:
    """Enum for Normalization Types"""
    LayerNorm = "layernorm"
    BatchNorm = "batchnorm"  # BN_{\mathbb{C}}
    BatchNormNaiv = "batchnormnaiv"  # BN_{\mathbb{R}^2}
    BatchNormVar = "batchnormvar"  # BN_{\mathbb{V}} using variance
    NoNorm = None


class CVKANLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8, grid_min = -2, grid_max = 2, rho=1, use_norm=Norms.BatchNorm, csilu_type="complex_weight"):
        """
        :param input_dim: input dimension size of Layer (Layer Width)
        :param output_dim: output dimension size of Layer (next Layer's Width)
        :param num_grids: number of grid points ***per dimension***; thus num_grids * num_grids gridpoints in total
        :param grid_min: left limit of grid
        :param grid_max: right limit of grid
        :param rho: rho for use in RBF (default rho=1)
        :param use_norm: which Normalization scheme to use
        :param csilu_type: the kind of CSiLU to use ('complex_weight' or 'real_weights')
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.csilu_type = csilu_type
        self.use_norm = use_norm
        # initialize Norm instance corresponding to self.use_norm
        if self.use_norm == Norms.LayerNorm:
            self.norm = Complex_LayerNorm()
        elif self.use_norm == Norms.BatchNorm:
            self.norm = Complex_BatchNorm(num_channel=output_dim)
        elif self.use_norm == Norms.BatchNormNaiv:
            self.norm = Complex_BatchNorm_naiv(num_channel=output_dim)
        elif self.use_norm == Norms.BatchNormVar:
            self.norm = Complex_BatchNorm_var(num_channel=output_dim)
        elif self.use_norm == Norms.NoNorm:
            self.norm = None
        else:
            raise NotImplementedError()
        # create grid points 2D array
        real = torch.linspace(grid_min, grid_max, num_grids)
        real = real.unsqueeze(1).expand(num_grids, num_grids)

        imag = torch.linspace(grid_min, grid_max, num_grids)
        imag = imag.unsqueeze(0).expand(num_grids, num_grids)
        # make it complex-valued from real and imaginary parts
        grid = torch.complex(real, imag)
        # grid is a non-trainable Parameter
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.rho = rho
        # weights for each RBF centered around the grid points
        self.realweights = torch.nn.Parameter(torch.randn(size=(input_dim, output_dim, num_grids, num_grids)), requires_grad=True)
        self.complexweights = torch.nn.Parameter(torch.randn(size=(input_dim, output_dim, num_grids, num_grids)), requires_grad=True)
        # initialize CSiLU weight to use based on selected csilu_type
        if self.csilu_type == "complex_weight":
            self.silu_weight = torch.nn.Parameter(torch.ones(size=(self.input_dim, self.output_dim), dtype=torch.complex64), requires_grad=True)
        elif self.csilu_type == "real_weights":
            self.silu_weight = torch.nn.Parameter(torch.ones(size=(self.input_dim, self.output_dim, 2), dtype=torch.float32), requires_grad=True)
        else:
            raise NotImplementedError()
        # add complex-valued bias to CSiLU
        self.silu_bias = torch.nn.Parameter(torch.zeros(size=(input_dim, output_dim), dtype=torch.complex64), requires_grad=True)


    def forward(self, x):
        # x has shape BATCH x Input-Dim
        assert len(x.shape) == 2 and x.shape[1] == self.input_dim, f"Wrong Input Dimension! Got {x.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}]"
        # grid has shape num_grids x num_grids
        # transform x to shape BATCH x Input-Dim x num_grids x num_grids
        x_expanded = x.unsqueeze(-1).unsqueeze(-1).expand(x.shape + (self.num_grids, self.num_grids,))
        # apply RBF on x (centered around each grid point)
        result = torch.exp(-(torch.abs(x_expanded - self.grid)) ** 2 / self.rho)
        # u and v are the two grid dimensions
        # i and o are input and output indices within layer layer_idx and layer_ix+1
        result_real = torch.einsum("biuv,iouv->bo", result, self.realweights)
        result_imag = torch.einsum("biuv,iouv->bo", result, self.complexweights)
        # construct complex number out of real and imaginary parts again
        result_complex = torch.complex(result_real, result_imag)
        assert result_complex.shape[1] == self.output_dim, f"Wrong Output Dimension! Got {result_complex.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}]"
        # kind of CSiLU used
        if self.csilu_type == "complex_weight":
            silu_value = torch.einsum("io,bi->bio",self.silu_weight, complex_silu_complexweight(x))
        elif self.csilu_type == "real_weights":
            silu_value_raw = complex_silu_realweights(x)
            silu_value = torch.complex(torch.einsum("io,bi->bio",self.silu_weight[:,:,0], silu_value_raw[0]),
                                       torch.einsum("io,bi->bio",self.silu_weight[:,:,1], silu_value_raw[1]))
        # Add complex CSiLU bias
        silu_value += self.silu_bias
        silu_value = torch.einsum("bio->bo", silu_value)
        result_complex = result_complex + silu_value
        # potentially apply Normalization
        if self.use_norm is not None:
            result_complex = self.norm(result_complex)
        return result_complex

    def to(self, device):
        super().to(device)
        # move grid to the right device
        self.grid = self.grid.to(device)
        if self.use_norm is not None:  # and Norm as well
            self.norm = self.norm.to(device)
class CVKAN(torch.nn.Module):
    def __init__(self,
                 layers_hidden: List[int],
                 num_grids: int = 8,
                 rho=1,
                 use_norm=Norms.BatchNorm,
                 grid_mins = -2,
                 grid_maxs = 2,
                 csilu_type = "complex_weight"):
        """
        :param layers_hidden: List with Layer Sizes (i.e. [1,5,3,1] for a 1x5x3x1 CVKAN)
        :param num_grids: Number of Grid Points ***per dimension*** (so in total num_grids * num_grids gridpoints)
        :param rho: rho for RBF (default rho=1)
        :param use_norm: which Normalization scheme to use. Normalization is applied AFTER every layer except the last
        :param grid_mins: left limit of grid
        :param grid_maxs: right limit of grid
        :param csilu_type: type of CSiLU to use ('complex_weight' or 'real_weights')
        """
        super().__init__()
        # convert grid limits to list if not already is list (limits for each layer independently)
        if not type(grid_mins) == list:
            grid_mins = [grid_mins] * len(layers_hidden)
        if not type(grid_maxs) == list:
            grid_maxs = [grid_maxs] * len(layers_hidden)
        self.layers_hidden = layers_hidden
        self.num_grids = num_grids
        self.rho = rho
        self.use_norm = use_norm
        self.csilu_type = csilu_type
        # convert csilu_type to list (each layer could get it's own CSiLU type)
        if type(self.use_norm) != list:
            self.use_norm = [self.use_norm] * (len(layers_hidden) - 1)
        else:
            assert len(self.use_norm) == len(self.layers_hidden)
        # lambdas to calculate if Layer i should have Normalization applied after it
        is_last_layer = lambda i: i >= len(self.layers_hidden) - 2
        norm_to_use = lambda i: self.use_norm[i] if not is_last_layer(i) else Norms.NoNorm
        # Array with Normalization schemes to use after every layer
        self.use_norm = [norm_to_use(i) for i in range(len(layers_hidden)-1)]
        # stack Layers into a ModuleList
        self.layers = torch.nn.ModuleList([CVKANLayer(input_dim=layers_hidden[i], output_dim=layers_hidden[i+1],
                                                      num_grids=num_grids, grid_min=grid_mins[i], grid_max=grid_maxs[i],
                                                      rho=self.rho, use_norm=norm_to_use(i),
                                                      csilu_type=self.csilu_type) for i in range(len(layers_hidden) - 1)])
    def forward(self, x):
        # make sure x is batched
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        # find mins over all samples within batch
        mins_per_channel = torch.minimum(torch.amin(x.real, dim=0), torch.amin(x.imag, dim=0))
        maxs_per_channel = torch.maximum(torch.amax(x.real, dim=0), torch.amax(x.imag, dim=0))
        # make sure first Layer's grid limits aren't overstepped. This would indicate a problem with dataset
        # normalization (which must be done before entering the data into the model!)
        assert (mins_per_channel >= self.layers[0].grid_min).all() and (maxs_per_channel <= self.layers[0].grid_max).all(), "Input data does not fall completely within the grid range of the first layer. Please normalize the data!"
        # feed data through the layers
        for layer in self.layers:
            x = layer(x)
        return x
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
