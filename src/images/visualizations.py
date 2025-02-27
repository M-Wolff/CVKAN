"""
File: visualizations.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Create the visualizations and figures used in our paper
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics

from src.cvkan.experiments.knot_dataset import load_knot_dataset
from src.cvkan.models.CVKAN import Norms
from src.cvkan.models.wrapper.CVKANWrapper import CVKANWrapper
from src.cvkan.train.train_loop import train_kans
from src.cvkan.utils.dataloading.create_complex_dataset import create_complex_dataset
from src.cvkan.utils.dataloading.csv_dataloader import CSVDataset
from src.cvkan.utils.eval_model import plot_confusion_matrix
from src.cvkan.utils.explain_kan import KANExplainer
from src.cvkan.utils.loss_functions import MSE, MAE
from src.cvkan.utils.plotting.plot_kan import KANPlot
import src.cvkan.utils.plotting.cplotting_tools as cplt
matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

def rbfs():
    """Create figure of real-valued RBFs forming learnable activation function (images/RBF_weighted_sum.svg)"""
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(12,10))
    h = 1
    grid_points = np.array([[-2], [0], [2]])
    weights = [[1.3],[-1.2],[0.7]]
    xs = np.arange(-5, 5, 0.01)
    rbfs = np.exp(-(xs - grid_points)**2 / (2*h**2))
    rbfs = weights * rbfs
    plt.plot(xs, rbfs[0,:], color="red", linewidth=0.5, label="RBF at $g_0=-2$ with $w_0=1.3$")
    plt.plot(xs, rbfs[1, :], color="blue", linewidth=0.5, label="RBF at $g_1=0$ with $w_1=-1.2$")
    plt.plot(xs, rbfs[2, :], color="green", linewidth=0.5, label="RBF at $g_2=2$ with $w_2=0.7$")
    plt.plot(xs, np.sum(rbfs, axis=0), color="black", linewidth=3, label="$\Phi(x)$ result of weighted summation")
    plt.plot([-2,-2],[-2.5,1.5],color="black", linewidth=1, linestyle="dashed")
    plt.plot([2, 2], [-2.5, 1.5], color="black", linewidth=1, linestyle="dashed")
    plt.xlim([-5, +5])
    plt.ylim([-2.5,1.5])
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()

    #plt.show()
    plt.savefig('images/RBF_weighted_sum.svg', bbox_inches='tight', transparent=True)
def rbf_hats():
    """Create figure of complex-valued RBFs (images/CRBF_hats.png)"""
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    h = 1
    num_grids = 3
    grid_min, grid_max, num_pts = -2, 2, 200

    # create meshgrid for real and imaginary parts of gridpoints
    reals_g, imags_g = np.meshgrid(np.linspace(grid_min, grid_max, num_grids),
                               np.linspace(grid_min, grid_max, num_grids))
    grid = torch.complex(torch.from_numpy(reals_g), torch.from_numpy(imags_g)).to(torch.complex64)

    # create meshgrid for real and imaginary parts of inputs
    reals_i, imags_i = np.meshgrid(np.linspace(-5, 5, num_pts),
                               np.linspace(-5, 5, num_pts))
    inputs = torch.complex(torch.from_numpy(reals_i), torch.from_numpy(imags_i)).to(torch.complex64)
    inputs = inputs.flatten()

    # grid has shape num_grids x num_grids
    # transform x to shape BATCH x Input-Dim x num_grids x num_grids
    x = inputs.unsqueeze(-1).unsqueeze(-1).expand(
        inputs.shape + (num_grids, num_grids))
    # apply RBF
    result = torch.exp(-(torch.abs(x - grid)) ** 2 / (2*h**2))
    result = result.reshape(num_pts, num_pts, num_grids, num_grids)
    # colors for the 9 RBF spikes / hats
    colors = [["tab:blue", "tab:orange", "tab:green"], ["tab:red", "tab:purple", "tab:brown"], ["tab:pink", "tab:olive", "tab:cyan"]]

    # iterate over first grid dimension
    for u in (range(num_grids)):
        # iterate over second grid dimension
        for v in (range(num_grids)):
            # if function value is kess than 0.05, then set it to NaN (limit spikes / hats to certain locations;
            # otherwise the functions would continue forever and all overlap weirdly in the plot)
            for i1 in range(num_pts):
                for i2 in range(num_pts):
                    if result[i1, i2, u, v] <= 0.05:
                        result[i1, i2, u, v] = torch.nan

            # plot spike / hat for gridpoint (u,v)
            ax.plot_surface(reals_i, imags_i, result[:,:,u,v], color=colors[u][v], alpha=0.7)
    # create black plane at zero
    ax.plot_surface(reals_i, imags_i, np.zeros_like(reals_i), color="black", alpha=0.7)
    ax.set_xlabel("$\Re(x)$", labelpad=10)
    ax.set_ylabel("$\Im(x)$", labelpad=10)
    ax.set_zlabel("$RBF_\mathbb{C}(x)$", labelpad=10)
    #plt.show()
    plt.savefig('images/CRBF_hats.png', transparent=True, bbox_inches='tight', pad_inches=0.5)

def plot_kan_square_square():
    """
    Plot model for CVKAN trained on $f(z1, z2) = (z1^2 + z2^2)^2$.
    Produced plot is images/cvkan_sqsq_plot.png
    """
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    mse_loss = MSE()
    mae_loss = MAE()
    # define MSE and MAE losses
    loss_fns = dict()
    loss_fns["mse"] = mse_loss
    loss_fns["mae"] = mae_loss

    # lambda for $f(z1, z2) = (z1^2 + z2^2)^2$
    sqsq = lambda x: (x[:, [0]] ** 2 + x[:, [1]] ** 2)**2
    # create dataset
    dataset = create_complex_dataset(sqsq, 2, [-1, 1], train_num=10000, test_num=5000, device="cuda")
    dataset = CSVDataset(dataset, input_vars=["$z_1$", "$z_2$"], output_vars=["$(z_1^2 + z_2^2)^2$"], categorical_vars=[])
    # create model and explainer
    model = CVKANWrapper(layers_hidden=[2,1,1], num_grids=8, rho=1, use_norm=Norms.NoNorm)
    model.to("cuda")
    kan_explainer = KANExplainer(model, samples=dataset.data["test_input"].to("cuda"), method="pykan")
    # train the model to sufficient precision (5000 epochs)
    train_kans(model, dataset=dataset, loss_fn_backprop=mse_loss, loss_fns=loss_fns, device="cuda", batch_size=2000,logging_interval=100,add_softmax_lastlayer=False,epochs=5000,last_layer_output_real=False, sparsify=True, kan_explainer=kan_explainer)

    # plot the model
    kan_plotter = KANPlot(model,kan_explainer=kan_explainer, input_featurenames=dataset.input_varnames, output_names=dataset.output_varnames, complex_valued=True, plot_options=None)
    kan_plotter.plot_all()
def plot_kan_knot():
    """
    Plot model for CVKAN trained on knot dataset.
    Produced plot is images/cvkan_knot_relevances.png
    as well as images/cvkan_knot_confusionmatrix.svg
    """
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    matplotlib.rc('font', **font)
    # define CE-Loss and accuracy
    crossentropy_loss = torch.nn.CrossEntropyLoss()
    loss_fns = dict()
    loss_fns["cross_entropy"] = crossentropy_loss
    loss_fns["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=14).to("cuda")
    # load knot dataset
    dataset = load_knot_dataset(complex_dataset=True)
    # create model and explainer
    model = CVKANWrapper(layers_hidden=[15,1,14], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
    model.to("cuda")
    kan_explainer = KANExplainer(model, samples=dataset.data["test_input"], method="pykan")
    # train CVKAN model
    train_kans(model, dataset=dataset, loss_fn_backprop=crossentropy_loss, loss_fns=loss_fns, device="cuda", batch_size=10000,
               logging_interval=10, add_softmax_lastlayer=True, epochs=200, last_layer_output_real=True, sparsify=True, kan_explainer=kan_explainer)
    # plot the model
    kan_plotter = KANPlot(model, kan_explainer=kan_explainer, input_featurenames=dataset.input_varnames,
                          output_names=dataset.output_varnames, complex_valued=True, plot_options=None)
    kan_plotter.plot_all()
    model.to("cuda")
    # plot confusionmatrix. Labelmapping converts class indices to class names. Highly underrepresented labels
    # have been removed as they are very unlikely (but not impossible) to show up in test split
    labelmapping = [-12, -10, -8, -6,-4,-2,0,2,4,6,8,10,12]  # removed 14
    plot_confusion_matrix(pred=model(dataset.data["test_input"].to("cuda")), gt=dataset.data["test_label"].to("cuda"), labelmapping=labelmapping)
def plot_ideal_zsquared():
    """
    Plot an ideal $f(z)=z^2$ function
    """
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 24}
    matplotlib.rc('font', **font)
    # meshgrid for real and imaginary parts
    xs, ys = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    # convert to complex numbers
    zs = xs + 1j * ys
    # square the complex-valued numbers
    f = zs ** 2
    # plot it
    cplt.complex_plot3D(xs, ys, f, fontsize=24)
if __name__ == '__main__':
    #rbfs()
    #rbf_hats()
    plot_kan_square_square()
    #plot_ideal_zsquared()
    #plot_kan_knot()