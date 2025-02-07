"""
File: CV_LayerNorm.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Three different complex-valued BatchNorm approaches and one complex-valued LayerNorm
"""
import torch.nn as nn
import torch


class Complex_LayerNorm(nn.Module):  # this is an alternate implementation of Layernorm where everything is wrapped in one module, the split one is preferable for explainability methods

    def __init__(self, device='cuda'):

        super().__init__()

    def forward(self, input):

        ev = torch.unsqueeze(torch.mean(input, dim=1), dim=1)

        cov_shape = [3, input.shape[0], 1]
        cov_m = torch.zeros(cov_shape, device=input.device)
        cov_m[0] = torch.unsqueeze(torch.var(input.real, dim=1), dim=1) + 1e-5  # 1e-5 if variance 0
        cov_m[1] = torch.unsqueeze(torch.var(input.imag, dim=1), dim=1) + 1e-5  # 1e-5 if variance 0

        input = input - ev

        cov_m[2] = torch.unsqueeze(torch.mean(input.real * input.imag, dim=1), dim=1)  # cov(real, imag)

        cov_m = self.inv_sqrt_2x2(cov_m)

        input = self.mult_2x2_same_dim(input, cov_m)  # decorrelate input

        return input

    def inv_sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / (t * s) * torch.cat([input[:, 1] + s, input[:, 0] + s, -input[:, 2]], dim=0)

    def sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / t * torch.cat([input[:, 0] + s, input[:, 1] + s, input[:, 2]], dim=0)

    def mult_2x2_same_dim(self, input, mult):
        return mult[0] * input.real + mult[2] * input.imag + 1j * (mult[2] * input.real + mult[1] * input.imag)


class Complex_BatchNorm(nn.Module):
    def __init__(self, num_channel, affine=True, momentum=0.1, device='cuda'):
        super().__init__()
        self.register_buffer('running_mean', tensor=torch.zeros([1, num_channel], dtype=torch.complex64, requires_grad=False))
        self.register_buffer('running_cov', tensor=torch.Tensor([1, 1, 0]).repeat([1, num_channel]))
        self.device = device
        self.zeros_cpu = torch.zeros(5, device=device)
        self.zeros_gpu = torch.zeros(5, device=self.device)
        self.momentum = momentum
        self.first_run = True
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.cat([torch.ones([2, num_channel], dtype=torch.float32, device=self.device), torch.zeros([1, num_channel], dtype=torch.float32, device=self.device)], dim=0))
            self.bias = nn.Parameter(torch.zeros(num_channel, dtype=torch.complex64, device=self.device))

    def forward(self, input):
        if self.training:

            ev = torch.unsqueeze(torch.mean(input, dim=0), dim=0)

            with torch.no_grad():
                if not self.first_run:
                    self.running_mean = (ev * (1 - self.momentum) + self.running_mean * self.momentum).detach()
                else:
                    self.running_mean = ev.detach()

            cov_m = torch.zeros([3, input.shape[1]], device=input.device)
            cov_m[0] = torch.var(input.real, dim=0) + 1e-5
            cov_m[1] = torch.var(input.imag, dim=0) + 1e-5  # 1e-5 if variance 0

            input = input - ev
            cov_m[2] = torch.mean(input.real * input.imag, dim=0)

            cov_m = self.inv_sqrt_2x2(cov_m)
            with torch.no_grad():
                if not self.first_run:
                    self.running_cov = (cov_m * (1 - self.momentum) + self.running_cov * self.momentum).detach()  # note: running_cov is already sqrt and inv
                else:
                    self.running_cov = cov_m.detach()
                    self.first_run = False

        else:
            cov_m = self.running_cov.detach()  # cov is already saved as inv_sqrt
            ev = self.running_mean.detach()

            input = input - ev

        input = self.mult_2x2(input, cov_m)  # decorrelate input

        if self.affine:
            # weight_calc = torch.zeros_like(self.weight)
            weight_var_real = self.weight[0]**2  # ensures varR>0
            weight_var_imag = self.weight[1]**2  # ensures varI>0
            weight_cov_ri = torch.sqrt(weight_var_real * weight_var_imag) * (torch.sigmoid(self.weight[2]) * 2 - 1)  # ensures covRI**2 < varR * varI
            weight_calc = torch.cat([weight_var_real.unsqueeze(0), weight_var_imag.unsqueeze(0), weight_cov_ri.unsqueeze(0)], dim=0)

            input = self.mult_2x2(input, weight_calc)
            input = input + self.bias

        return input

    def mult_2x2(self, input, mult):
        mult = torch.unsqueeze(mult, dim=0)
        return mult[:, 0] * input.real + mult[:, 2] * input.imag + 1j * (mult[:, 2] * input.real + mult[:, 1] * input.imag)

    def inv_sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / (t * s) * torch.cat([input[:, 1] + s, input[:, 0] + s, -input[:, 2]], dim=0)

    def sqrt_2x2(self, input):
        input = torch.unsqueeze(input, dim=0)
        s = torch.sqrt(input[:, 0] * input[:, 1] - input[:, 2] ** 2)
        t = torch.sqrt(input[:, 0] + input[:, 1] + 2 * s)
        return 1 / t * torch.cat([input[:, 0] + s, input[:, 1] + s, input[:, 2]], dim=0)


class Complex_BatchNorm_naiv(nn.Module):

    def __init__(self, num_channel, affine=True, eps=1e-05, momentum=0.1, device='cuda', dtype=torch.float32):
        super().__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.dtype = dtype
        self.affine = affine

        self.real_BatchNorm = nn.BatchNorm1d(self.num_channel, eps=self.eps, momentum=self.momentum, affine=affine, track_running_stats=True, device=self.device, dtype=self.dtype)
        self.imag_BatchNorm = nn.BatchNorm1d(self.num_channel, eps=self.eps, momentum=self.momentum, affine=affine, track_running_stats=True, device=self.device, dtype=self.dtype)

    def forward(self, input):
        return torch.complex(self.real_BatchNorm(input.real), self.imag_BatchNorm(input.imag))


class Complex_BatchNorm_var(nn.Module):
    def __init__(self, num_channel, affine=True, momentum=0.1, device='cuda'):
        super().__init__()
        self.register_buffer('running_mean', tensor=torch.zeros([1, num_channel], dtype=torch.complex64, requires_grad=False))
        self.register_buffer('running_var', tensor=torch.ones([1, num_channel], dtype=torch.complex64, requires_grad=False))
        self.momentum = momentum
        self.first_run = True
        self.device = device
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channel, dtype=torch.float32, device=self.device))
            self.bias = nn.Parameter(torch.zeros(num_channel, dtype=torch.complex64, device=self.device))
            # with torch.no_grad():
            #     self.weight.fill_(1)
            #     self.bias.fill_(0)

    def forward(self, input):

        if self.training:

            ev = torch.unsqueeze(torch.mean(input, dim=0), dim=0)

            with torch.no_grad():
                if not self.first_run:
                    self.running_mean = (ev * (1 - self.momentum) + self.running_mean * self.momentum).detach()
                else:
                    self.running_mean = ev.detach()

            var = torch.var(input, dim=0).unsqueeze(0)

            with torch.no_grad():
                if not self.first_run:
                    self.running_var = (var * (1 - self.momentum) + self.running_var * self.momentum).detach()
                else:
                    self.running_var = var.detach()
                    self.first_run = False

        else:
            var = self.running_var.detach()
            ev = self.running_mean.detach()

        input = (input - ev) / torch.sqrt(var)
        if self.affine:
            input = input * self.weight + self.bias

        return input