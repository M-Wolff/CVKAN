"""
File: fit_formulas.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Experiments for Function Fitting (physically meaningful circuit & holography
             as well as arbitrary simple formulae)
"""

import argparse
import torch
from kan import create_dataset

from ..experiments.run_crossval import run_crossval
from ..models.CVKAN import Norms
from ..models.FastKAN import FastKAN
from ..models.wrapper.CVKANWrapper import CVKANWrapper
from ..models.wrapper.PyKANWrapper import PyKANWrapper
from ..utils.dataloading.create_complex_dataset import create_complex_dataset
from ..utils.dataloading.csv_dataloader import CSVDataset
from ..utils.loss_functions import MSE, MAE

mse_loss = MSE()
mae_loss = MAE()

loss_fns = dict()
loss_fns["mse"] = mse_loss
loss_fns["mae"] = mae_loss

def convert_complex_dataset_to_real(dataset: CSVDataset):
    """Converts a complex dataset to a real-valued dataset by doubling input and output dimension (one real number for
    real and imaginary part each)"""
    if dataset.categorical_vars:
        raise NotImplementedError("complex dataset can't contain categorical vars!")
    num_vars_complex = len(dataset.input_varnames)
    num_vars_real = 2 * num_vars_complex
    num_outputs_complex = len(dataset.output_varnames)
    num_outputs_real = 2 * num_outputs_complex
    num_samples_train, num_samples_test = dataset.get_train_test_size()

    train_input = torch.zeros((num_samples_train, num_vars_real), dtype=torch.float32)
    test_input = torch.zeros((num_samples_test, num_vars_real), dtype=torch.float32)
    train_label = torch.zeros((num_samples_train, num_outputs_real), dtype=torch.float32)
    test_label = torch.zeros((num_samples_test, num_outputs_real), dtype=torch.float32)

    # make sure varnames are also doubled for later plotting
    input_varnames_real = []
    output_varnames_real = []

    for in_feature in range(num_vars_complex):
        train_input[:, 2*in_feature] = dataset.data["train_input"][:, in_feature].real
        train_input[:, 2*in_feature+1] = dataset.data["train_input"][:, in_feature].imag

        if num_samples_test > 0:
            test_input[:, 2*in_feature] = dataset.data["test_input"][:, in_feature].real
            test_input[:, 2*in_feature+1] = dataset.data["test_input"][:, in_feature].imag

        input_varnames_real.append(dataset.input_varnames[in_feature] + ".real")
        input_varnames_real.append(dataset.input_varnames[in_feature] + ".imag")
    for out_feature in range(num_outputs_complex):
        train_label[:, 2*out_feature] = dataset.data["train_label"][:, out_feature].real
        train_label[:, 2*out_feature+1] = dataset.data["train_label"][:, out_feature].imag

        if num_samples_test > 0:
            test_label[:, 2*out_feature] = dataset.data["test_label"][:, out_feature].real
            test_label[:, 2*out_feature+1] = dataset.data["test_label"][:, out_feature].imag

        output_varnames_real.append(dataset.output_varnames[out_feature] + ".real")
        output_varnames_real.append(dataset.output_varnames[out_feature] + ".imag")
    # build a dictionary out of the now real-valued datapoints
    realdata_dict = dict()
    realdata_dict['train_input'] = train_input
    realdata_dict['train_label'] = train_label
    realdata_dict['test_input'] = test_input
    realdata_dict['test_label'] = test_label

    # create a CVDataset object from this dict
    dataset_real = CSVDataset(realdata_dict, input_vars=input_varnames_real, output_vars=output_varnames_real, categorical_vars=[])
    return dataset_real

def run_experiments_physics(run_dataset, run_model):
    """Main method to run experiments on physically meaningul formula fitting (circuit & holography)"""
    _num_samples = 100000
    _dataset_name_suffix = "_100k"  # differentiate runs on 100k and 5k samples
    loss_fn_backprop = loss_fns["mse"]

    # holography formula
    holography = lambda x: torch.abs(x[:, [0]] + x[:, [1]]) ** 2 * x[:, [2]]
    # only generate train samples and no test samples because run_crossval later splits them into k folds
    dataset_holography_c = create_complex_dataset(holography, ranges=[-2,2], n_var=3, train_num=_num_samples, test_num=0)
    dataset_holography_c = CSVDataset(dataset_holography_c, input_vars=["Er1", "E0", "Er2"], output_vars=["holography"],
                                      categorical_vars=[])
    # create real-valued holography dataset from complex one
    dataset_holography_r = convert_complex_dataset_to_real(dataset_holography_c)

    # circuit dataset formula (complex and real separately, as most variables are not complex to begin with so we can't
    # easily convert complex dataset to real one)
    # variable indices: 0: U_g, 1: R_g, 2: R_l, 3: w, 4: L, 5: C
    circuit_complex_inputs = lambda x: x[:, [0]] / (1 + x[:, [1]].real/x[:, [2]].real - x[:, [3]].real**2 * x[:, [4]].real * x[:, [5]].real + 1j * x[:, [3]].real * (x[:, [4]].real/ x[:, [2]].real + x[:, [1]].real * x[:, [5]].real))
    # variable indices: 0: U_g.real, 1: U_g.imag, 2: R_g, 3: R_l, 4: w, 5: L, 6: C
    circuit_real_inputs_helper = lambda x: (torch.complex(x[:, [0]], x[:, [1]]) /
                                     torch.complex(1+x[:, [2]]/x[:, [3]] - x[:, [4]]**2 * x[:, [5]] * x[:, [6]], x[:, [4]] * (x[:, [5]] / x[:, [3]] + x[:, [2]] * x[:, [6]])))
    circuit_real_inputs = lambda x: torch.stack((circuit_real_inputs_helper(x).real, circuit_real_inputs_helper(x).imag), dim=1).squeeze()
    # only generate train samples and no test samples because run_crossval later splits them into k folds
    dataset_circuit_c = create_complex_dataset(circuit_complex_inputs, ranges=[-2, 2], n_var=6, train_num=_num_samples, test_num=0)
    # zero out imaginary parts of real-input-variables in complex dataset
    for var_idx in [1,2,3,4,5]:
        dataset_circuit_c["train_input"][:, var_idx] = torch.complex(dataset_circuit_c["train_input"][:, var_idx].real, 0*dataset_circuit_c["train_input"][:, var_idx].imag)
    dataset_circuit_c = CSVDataset(dataset_circuit_c, input_vars=["U_g", "R_g", "R_l", "w", "L", "C"], output_vars=["U_{rl}"], categorical_vars=[])
    # only generate train samples and no test samples because run_crossval later splits them into k folds
    dataset_circuit_r = create_dataset(circuit_real_inputs, ranges=[-2, 2], n_var=7, train_num=_num_samples, test_num=0)
    dataset_circuit_r = CSVDataset(dataset_circuit_r, input_vars=["U_g.real","U_g.imag", "R_g", "R_l", "w", "L", "C"],
                                   output_vars=["U_{rl}.real", "U_{rl}.imag"], categorical_vars=[])

    # check which model and dataset to run
    run_models = [False] * 3
    if run_model == "all":
        run_models = [True] * 3
    elif run_model == "pykan":
        run_models[0] = True
    elif run_model == "fastkan":
        run_models[1] = True
    elif run_model == "cvkan":
        run_models[2] = True

    run_datasets = [False] * 2
    if run_dataset == "all":
        run_datasets = [True] * 2
    elif run_dataset == "holography":
        run_datasets[0] = True
    elif run_dataset == "circuit":
        run_datasets[1] = True

    if run_datasets[0]:  # holography
        for arch in [[6,10,5,3,2], [6,1,2], [6,5,2], [6,10,2]]:
            if run_models[0]:
                # important to put batch_size = -1 for PyKAN as they do batching differently (randomly sampling
                # batch_size samples per step)
                pykan = PyKANWrapper(layers_hidden=arch, num_grids=64, update_grid=True, device="cuda")
                run_crossval(pykan, dataset_holography_r, dataset_name="ph_holo_r"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop,
                             loss_fns=loss_fns,
                             batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
            if run_models[1]:
                fastkan = FastKAN(layers_hidden=arch, num_grids=64, use_batchnorm=True)
                run_crossval(fastkan, dataset_holography_r, dataset_name="ph_holo_r"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                             batch_size=10000, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        for arch in [[3,1], [3,1,1], [3,3,1], [3,10,1], [3,10,3,1], [3,10,5,3,1]]:
            if run_models[2]:
                cvkan = CVKANWrapper(layers_hidden=arch, num_grids=8, rho=1, use_norm=Norms.BatchNorm)
                run_crossval(cvkan, dataset_holography_c, dataset_name="ph_holo_c"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop,
                             loss_fns=loss_fns,
                             batch_size=10000, add_softmax_lastlayer=False, epochs=1000,
                             convert_model_output_to_real=False)
    if run_datasets[1]:  # circuit
        for arch in [[7,1,2], [7,5,2], [7,10,2], [7,10,5,3,2]]:
            if run_models[0]:
                # important to put batch_size = -1 for PyKAN as they do batching differently (randomly sampling
                # batch_size samples per step)
                pykan = PyKANWrapper(layers_hidden=arch, num_grids=64, update_grid=True, device="cuda")
                run_crossval(pykan, dataset_circuit_r, dataset_name="ph_circuit_r"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop,
                             loss_fns=loss_fns,
                             batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
            if run_models[1]:
                fastkan = FastKAN(layers_hidden=arch, num_grids=64, use_batchnorm=True)
                run_crossval(fastkan, dataset_circuit_r, dataset_name="ph_circuit_r"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                             batch_size=10000, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        for arch in [[6,1], [6,1,1], [6,3,1], [6,10,1], [6,10,3,1], [6,10,5,3,1]]:
            if run_models[2]:
                cvkan = CVKANWrapper(layers_hidden=arch, num_grids=8, rho=1, use_norm=Norms.BatchNorm)
                run_crossval(cvkan, dataset_circuit_c, dataset_name="ph_circuit_c"+_dataset_name_suffix, loss_fn_backprop=loss_fn_backprop,
                             loss_fns=loss_fns,
                             batch_size=10000, add_softmax_lastlayer=False, epochs=1000,
                             convert_model_output_to_real=False)




def run_experiments_funcfitting(run_dataset = "all", run_model="all"):
    """Main method to run experiments on arbitrary simple formula fitting (z^2, sin(z), z_1*z_2, (z_1^2 + z_2^2)^2)"""
    loss_fn_backprop = loss_fns["mse"]
    sq = lambda x: x[:, [0]]**2
    sqsq = lambda x: ((x[:, [0]]) ** 2 + x[:, [1]] ** 2) ** 2
    mult = lambda x: (x[:, [0]]) * x[:, [1]]
    sinus = lambda x: torch.sin(x[:,[0]])
    run_datasets = [False] * 4
    if run_dataset == "all":
        run_datasets = [True] * 4
    elif run_dataset == "square":
        run_datasets[0] = True
    elif run_dataset == "squaresquare":
        run_datasets[1] = True
    elif run_dataset == "mult":
        run_datasets[2] = True
    elif run_dataset == "sinus":
        run_datasets[3] = True

    run_models = [False] * 3
    if run_model == "all":
        run_models = [True] * 3
    elif run_model == "pykan":
        run_models[0] = True
    elif run_model == "fastkan":
        run_models[1] = True
    elif run_model == "cvkan":
        run_models[2] = True

    # only generate train samples and no test samples because run_crossval later splits them into k folds

    dataset_sq_c = create_complex_dataset(sq, ranges=[-2,2], n_var=1, train_num=5000, test_num=0)
    dataset_sq_c = CSVDataset(dataset_sq_c, input_vars=["z"], output_vars=["z^2"], categorical_vars=[])
    dataset_sq_r = convert_complex_dataset_to_real(dataset_sq_c)

    dataset_sqsq_c = create_complex_dataset(sqsq, ranges=[-2,2], n_var=2, train_num=5000, test_num=0)
    dataset_sqsq_c = CSVDataset(dataset_sqsq_c, input_vars=["z_1", "z_2"], output_vars=["(z_1^2 + z_2^2)^2"], categorical_vars=[])
    dataset_sqsq_r = convert_complex_dataset_to_real(dataset_sqsq_c)

    dataset_mult_c = create_complex_dataset(mult, ranges=[-2, 2], n_var=2, train_num=5000, test_num=0)
    dataset_mult_c = CSVDataset(dataset_mult_c, input_vars=["z_1", "z_2"], output_vars=["z_1 * z_2"],
                            categorical_vars=[])
    dataset_mult_r = convert_complex_dataset_to_real(dataset_mult_c)

    dataset_sin_c = create_complex_dataset(sinus, ranges=[-2,2], n_var=1, train_num=5000, test_num=0)
    dataset_sin_c = CSVDataset(dataset_sin_c, input_vars=["z"], output_vars=["sin(z)"], categorical_vars=[])
    dataset_sin_r = convert_complex_dataset_to_real(dataset_sin_c)

    # Square Dataset = z**2
    if run_datasets[0]:
        if run_models[0]:
            pykan = PyKANWrapper(layers_hidden=[2, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sq_r, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            pykan = PyKANWrapper(layers_hidden=[2, 3, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sq_r, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[1]:
            fastkan = FastKAN(layers_hidden=[2, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sq_r, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            fastkan = FastKAN(layers_hidden=[2, 3, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sq_r, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[2]:
            cvkan = CVKANWrapper(layers_hidden=[1, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sq_c, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            cvkan = CVKANWrapper(layers_hidden=[1,2, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sq_c, dataset_name="ff_square", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

    # Square Square Dataset = (z_1**2 + z_2**2)**2
    if run_datasets[1]:
        if run_models[0]:
            pykan = PyKANWrapper(layers_hidden=[4, 2, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sqsq_r, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            pykan = PyKANWrapper(layers_hidden=[4, 6, 2, 3, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sqsq_r, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[1]:
            fastkan = FastKAN(layers_hidden=[4, 2, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sqsq_r, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            fastkan = FastKAN(layers_hidden=[4,6,2,3,2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sqsq_r, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[2]:
            cvkan = CVKANWrapper(layers_hidden=[2,1, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sqsq_c, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            cvkan = CVKANWrapper(layers_hidden=[2, 4, 2, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sqsq_c, dataset_name="ff_squaresquare", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

    # Mult Dataset = z_1 * z_2
    if run_datasets[2]:
        if run_models[0]:
            pykan = PyKANWrapper(layers_hidden=[4, 4, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_mult_r, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            pykan = PyKANWrapper(layers_hidden=[4, 8, 4, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_mult_r, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[1]:
            fastkan = FastKAN(layers_hidden=[4, 4, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_mult_r, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            # should be optimal fastkan able to compute z1 * z2
            fastkan = FastKAN(layers_hidden=[4,8,4,2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_mult_r, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[2]:
            cvkan = CVKANWrapper(layers_hidden=[2,2, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_mult_c, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            cvkan = CVKANWrapper(layers_hidden=[2, 4, 2, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_mult_c, dataset_name="ff_mult", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

    # sin Dataset = sin(z)
    if run_datasets[3]:
        if run_models[0]:
            pykan = PyKANWrapper(layers_hidden=[2, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sin_r, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            pykan = PyKANWrapper(layers_hidden=[2, 4, 4, 2], num_grids=64, update_grid=True, device="cuda")
            run_crossval(pykan, dataset_sin_r, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop,
                         loss_fns=loss_fns,
                         batch_size=-1, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[1]:
            fastkan = FastKAN(layers_hidden=[2, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sin_r, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            # should be correct network size (sin of complex number = cosh ...)
            fastkan = FastKAN(layers_hidden=[2, 4,4, 2], num_grids=64, use_batchnorm=True)
            run_crossval(fastkan, dataset_sin_r, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)
        if run_models[2]:
            cvkan = CVKANWrapper(layers_hidden=[1, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sin_c, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)

            cvkan = CVKANWrapper(layers_hidden=[1,2, 1], num_grids=8, rho=1, use_norm=Norms.BatchNorm)
            run_crossval(cvkan, dataset_sin_c, dataset_name="ff_sin", loss_fn_backprop=loss_fn_backprop, loss_fns=loss_fns,
                         batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default="all", type=str)
    parser.add_argument('--model', nargs='?', default="all", type=str)
    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()
    print("Running Function Fitting Eperiments for Dataset ", args.dataset, " and Model ", args.model, " and Task ", args.task)
    if args.task == "funcfit":
        run_experiments_funcfitting(run_dataset=args.dataset, run_model=args.model)
    if args.task == "physics":
        run_experiments_physics(run_dataset=args.dataset, run_model=args.model)
