"""
File: cplot.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Experiments with plotting standard complex-valued functions (i.e. z^2)
"""
import matplotlib
import cplotting_tools as cplt
import numpy as np

matplotlib.use('TkAgg')  # use a Backend that supports interactive elements

# create meshgrid for range [-2, 2] x [-2, 2]
xs, ys = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))

# convert meshgrid to complex-valued numbers (real and imaginary parts)
zs = xs + 1j*ys

# calculate result
f = zs**2
#f = np.sin(zs)
#f = np.exp(3*zs)

# plot result
cplt.complex_plot3D(xs, ys, f)
cplt.plot_re_im(xs, ys, f)
#cplt.domain_coloring(xs,ys,f,cmap="hsv")
