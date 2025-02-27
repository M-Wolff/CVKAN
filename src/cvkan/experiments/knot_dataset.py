"""
File: knot_dataset.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Experiments on the Knot Dataset (download dataset.csv from google storage
             gs://maths_conjectures/knot_theory/knot_theory_invariants.csv
             and also see their Jupyter Notebook:
             https://github.com/google-deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb
             )
"""

from pathlib import Path
import pandas as pd
import torch
import torchmetrics

from ..experiments.run_crossval import run_crossval
from ..models.CVKAN import Norms
from ..models.FastKAN import FastKAN
from ..models.wrapper.CVKANWrapper import CVKANWrapper
from ..models.wrapper.PyKANWrapper import PyKANWrapper
from ..utils.dataloading.csv_dataloader import CSVDataset

def load_knot_dataset(input_filename=Path("/home/m_wolf37/Datasets/knot_theory_invariants.csv"), train_test_split="70:30", complex_dataset=False):
    """
    Load Knot Dataset from .csv file
    :param input_filename: Path to dataset .csv file
    :param train_test_split: String specifying the share of train and test split as 'trainpercent:testpercent'
    :param complex_dataset: Whether return should be a complex-valued dataset (True) or real-valued (False)
    :return: Complex-valued or real-valued dataset
    """
    full_df = pd.read_csv(input_filename)
    full_df.drop(full_df.columns[0], axis=1, inplace=True)  # drop first column (id)
    # extract column names for input and output
    input_vars = list(full_df.columns[0:len(full_df.columns) - 1])
    # varnames for complex-valued dataset (abbreviated). Note that this list contains imag and real parts as
    # one single complex-valued variable
    input_vars_short_complex = ["adjoint_torsion_deg", "torsion_deg", "short_geodesic", "inject_rad", "chern_simons", "cusp_vol", "longit_translat", "merid_translat_c", "volume", "sym_0", "sym_d3", "sym_d4", "sym_d6", "sym_d8", "sym_Z/2+Z/2"]
    output_var = [full_df.columns[len(full_df.columns) - 1]]
    # load dataset (real valued)
    dataset = CSVDataset(full_df, input_vars=input_vars, output_vars=output_var, categorical_vars=output_var, train_test=train_test_split)
    # normalize dataset (to our fixed grid range of [-2, 2])
    dataset.normalize()

    if not complex_dataset:  # if interested in complex-valued knot dataset
        return dataset
    # otherwise: use this dataset to construct a complex-valued dataset
    num_train, num_test = dataset.data["train_input"].shape[0], dataset.data["test_input"].shape[0]
    num_complex_input_vars = len(dataset.input_varnames) - 2  # -2 because there are already 2 complex numbers in the dataset (split in Re and Im)
    dataset_complex = dict()
    dataset_complex["train_input"] = torch.zeros((num_train, num_complex_input_vars), dtype=torch.complex64)
    dataset_complex["test_input"] = torch.zeros((num_test, num_complex_input_vars), dtype=torch.complex64)

    # copy input features into complex dataset setting imaginary part to zero (if it doesnt exist)
    for idx_orig, idx_complex in [(0,0), (1,1), (4,3), (5,4), (6,5), (7,6), (10,8), (11,9), (12,10), (13,11), (14,12), (15,13), (16,14)]:
        print(idx_orig, idx_complex, dataset.data["train_input"].shape)
        dataset_complex["train_input"][:,idx_complex] = torch.complex(dataset.data["train_input"][:,idx_orig], torch.zeros_like(dataset.data["train_input"][:,idx_orig]))
        dataset_complex["test_input"][:,idx_complex] = torch.complex(dataset.data["test_input"][:,idx_orig], torch.zeros_like(dataset.data["test_input"][:,idx_orig]))

    # make complex number for "short geodesic"
    dataset_complex["train_input"][:, 2] = torch.complex(dataset.data["train_input"][:, 2], dataset.data["train_input"][:, 3])
    dataset_complex["test_input"][:, 2] = torch.complex(dataset.data["test_input"][:, 2], dataset.data["test_input"][:, 3])
    # make complex number for "meridinal translation" (in CSV imag and real are swapped...)
    dataset_complex["train_input"][:, 7] = torch.complex(dataset.data["train_input"][:, 9], dataset.data["train_input"][:, 8])
    dataset_complex["test_input"][:, 7] = torch.complex(dataset.data["test_input"][:, 9], dataset.data["test_input"][:, 8])

    # copy labels
    dataset_complex["train_label"] = dataset.data["train_label"]
    dataset_complex["test_label"] = dataset.data["test_label"]

    # change input vars
    input_vars[2] = "short_geodesic_complex"
    input_vars[8] = "meridinal_translation_complex"
    del input_vars[9]
    del input_vars[3]
    dataset_complex = CSVDataset(dataset_complex, input_vars=input_vars, output_vars=["c" + str(i) for i in range(14)], categorical_vars=[])

    dataset_complex.num_classes = dataset.num_classes
    dataset_complex.input_varnames = input_vars_short_complex
    return dataset_complex
def run_experiments():
    """Run the experiments on the Knot Dataset"""
    _DEVICE = torch.device("cuda")
    # load knot complex and real-valued with 100% train split
    # use 100% train split because run_crossval expects it this way and splits it into 5 non-overlapping folds
    knot_dataset_complex = load_knot_dataset(train_test_split="100:0", complex_dataset=True)
    knot_dataset_real = load_knot_dataset(train_test_split="100:0", complex_dataset=False)
    in_features_real = len(knot_dataset_real.input_varnames)
    in_features_complex = len(knot_dataset_complex.input_varnames)
    num_classes = len(knot_dataset_complex.output_varnames)

    crossentropy_loss = torch.nn.CrossEntropyLoss()

    loss_fns = dict()
    loss_fns["cross_entropy"] = crossentropy_loss
    loss_fns["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(_DEVICE)

    # Experiments for FastKAN and pyKAN
    for arch, use_batchnorm, num_grids in [([in_features_real, 1, num_classes], True, 8), ([in_features_real, 1, num_classes], True, 64), ([in_features_real, 2, num_classes], True, 8), ([in_features_real, 2, num_classes], True, 64)]:
        fastkan = FastKAN(layers_hidden=arch,
                          num_grids=num_grids, use_batchnorm=use_batchnorm, grid_mins=-2, grid_maxs=2)
        run_crossval(fastkan, knot_dataset_real, dataset_name="knot_r", loss_fn_backprop=crossentropy_loss, loss_fns=loss_fns, device=_DEVICE,
                     batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200, convert_model_output_to_real=False)
        pykan = PyKANWrapper(layers_hidden=arch, num_grids=num_grids, update_grid=use_batchnorm, device=_DEVICE, grid_range=[-2,2])
        run_crossval(pykan, knot_dataset_real, dataset_name="knot_r", loss_fn_backprop=loss_fns["cross_entropy"],
                     loss_fns=loss_fns,
                     batch_size=-1, add_softmax_lastlayer=True, epochs=200, convert_model_output_to_real=False, device=torch.device("cuda"))

    # Experiments for our CVKAN
    arch = [in_features_complex, 2, num_classes]
    num_grids = 8
    for batchnorm in [Norms.BatchNorm, Norms.BatchNormVar, Norms.BatchNormNaiv, Norms.NoNorm]:
        for zsilu_type in ["complex_weight", "real_weights"]:
            cvkan = CVKANWrapper(layers_hidden=arch, num_grids=num_grids, rho=1, use_norm=batchnorm, grid_mins=-2, grid_maxs=2, zsilu_type=zsilu_type)
            run_crossval(cvkan, knot_dataset_complex, dataset_name="knot_c", loss_fn_backprop=crossentropy_loss,
                         loss_fns=loss_fns, device=_DEVICE,
                         batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200, convert_model_output_to_real=True)


def train_knot_feature_subset():
    """Training on the Knot Dataset on the most important 3 or 7 features only
    as well as on the inverse set of every feature except the most important 3 or 7"""
    # use 100% train split because run_crossval expects it this way and splits it into 5 non-overlapping folds
    knot_dataset = load_knot_dataset(train_test_split="100:0", complex_dataset=True)
    # indices for features to train on
    indices_big = [1,2,3,5,6,7,8]
    indices_small = [2,6,7]
    indices_big_inverse = [0,4,9,10,11,12,13,14]
    indices_small_inverse = [0,1,3,4,5,8,9,10,11,12,13,14]

    num_classes = 14
    _DEVICE = torch.device("cuda")
    crossentropy_loss = torch.nn.CrossEntropyLoss()

    loss_fns = dict()
    loss_fns["cross_entropy"] = crossentropy_loss
    loss_fns["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(_DEVICE)
    # train only on a specific subset of input-features
    for indices in [indices_small, indices_big, indices_small_inverse, indices_big_inverse]:
        cvkan = CVKANWrapper(layers_hidden=[len(indices), 1, 14], num_grids=8, rho=1, use_norm=Norms.BatchNorm, grid_mins=-2, grid_maxs=2, zsilu_type="complex_weight")
        train_input = knot_dataset.data["train_input"][:,indices]
        # reduce dataset to the specified subset of features
        knot_dataset_complex_reduced = dict()
        knot_dataset_complex_reduced["train_input"] = train_input
        knot_dataset_complex_reduced["train_label"] = knot_dataset.data["train_label"]
        knot_dataset_complex_reduced["test_label"] = knot_dataset.data["test_label"]
        knot_dataset_complex_reduced["test_input"] = knot_dataset.data["test_label"]
        knot_dataset_complex_reduced = CSVDataset(knot_dataset_complex_reduced, input_vars=[knot_dataset.input_varnames[i] for i in indices], output_vars=knot_dataset.output_varnames, categorical_vars=[])

        run_crossval(cvkan, knot_dataset_complex_reduced, dataset_name="knot_c_"+str(indices), loss_fn_backprop=crossentropy_loss,
                     loss_fns=loss_fns, device=_DEVICE,
                     batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200,
                     convert_model_output_to_real=True)

if __name__ == "__main__":
    #run_experiments()
    train_knot_feature_subset()
