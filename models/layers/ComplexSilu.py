"""
File: ComplexSilu.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Two different variants of complex SiLU
"""
import torch


def complex_silu_complexweight(complex_input):
    """
    Applies SiLU on real and imaginary parts seperately, then combines both into a complex number and returns it
    :param complex_input: Input
    :return: SiLu(Input.real) + i*SiLU(Input.imag)
    """
    return torch.complex(torch.nn.functional.silu(complex_input.real, inplace=False), torch.nn.functional.silu(complex_input.imag, inplace=False))
def complex_silu_realweights(complex_input):
    """
    Calculates SiLU on real and imaginary parts seperately, returns result as a 2-tuple (so we can weight each element
    differently later)
    :param complex_input: Input
    :return: Tuple (SiLU(Input.real), SiLU(Input.imag))
    """
    return (torch.nn.functional.silu(complex_input.real, inplace=False), torch.nn.functional.silu(complex_input.imag, inplace=False))