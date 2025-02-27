"""
File: latex_table_creator.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Automatically create LaTeX Tables from results.json logfile
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

def get_dataset_name(json_data):
    """
    Returns dataset name
    :param json_data: JSON entry for current experiment
    :return: dataset name
    """
    res = json_data["dataset_name"]
    # convert internal names (i.e. ff_mult) to proper names
    replacement_dict = {"ff_mult": "$z_1*z_2$", "ff_sin": "$\sin (z)$", "ff_square": "$z^2$", "ff_squaresquare": "$(z_1^2 + z_2^2)^2$",
                        "ph_holo_r_100k": "Holography", "ph_circuit_r_100k": "Circuit", "ph_holo_c_100k": "Holography", "ph_circuit_c_100k": "Circuit"}
    return replacement_dict[res] if res in replacement_dict else res
def get_model_name(json_data):
    """
    Returns model name
    :param json_data: JSON entry for current experiment
    :return: model name
    """
    res = json_data["model_name"]
    # convert internal model names to more understandable names
    replacement_dict = {"FastKAN": "FastKAN", "CVKANWrapper": "$\C$KAN", "PyKANWrapper": "KAN"}
    return replacement_dict[res] if res in replacement_dict else res
def get_model_size(json_data):
    """
    Returns size of model (layer_0 x layer_1 x ... x layer_N) as a string
    :param json_data: JSON entry for current experiment
    :return: size of model (layer_0 x layer_1 x ... x layer_N)
    """
    return "$" + " {\\times} ".join([str(l) for l in json_data["functions"]]) + "$"
def get_model_params(json_data):
    """
    Returns number of trainable parameters
    :param json_data: JSON entry for current experiment
    :return: number of trainable parameters
    """
    return str(json_data["num_trainable_params"])
def get_model_normalization(json_data):
    """
    Returns normalization scheme used during current experiment
    :param json_data: JSON entry for current experiment
    :return: normalization scheme used
    """
    res = json_data["use_norm"]
    # assume that all layers used the same normalization scheme; thus just take the normalization scheme of first layer
    if type(res) == list:
        res = res[0]
    # replace internal names with something more understandable
    replacement_dict = {True: "BN", False : "no", "batchnorm": "$\\text{BN}_\\C$", "batchnormvar": "$\\text{BN}_\\mathbb{V}$", "batchnormnaiv": "$\\BN_{\\R^2}$", None: "no", "grid_update": "grid\_update"}
    return replacement_dict[res] if res in replacement_dict else res
def get_model_num_grids(json_data):
    """
    Returns number of gridpoints used
    :param json_data: JSON entry for current experiment
    :return: number of gridpoints used
    """
    return json_data["num_grids"]
def get_model_zsilu_type(json_data):
    """
    Returns type of complex-SiLU used during current experiment
    :param json_data: JSON entry for current experiment
    :return: type of complex-SiLU used
    """
    res = json_data["zsilu_type"]
    # replace internal names with something more understandable
    replacement_dict = {"complex_weight": "c", "real_weights": "r", None: "-"}
    return replacement_dict[res] if res in replacement_dict else res
def get_test_acc(json_data):
    """
    Returns accuracy on test-set averaged over all folds including standard deviation as a string
    :param json_data: JSON entry for current experiment
    :return: string for test accuracy in the form of '0.98 +- 0.02'
    """
    return ("$" +
            "{:.3f}".format(json_data["test_losses"]["mean"]["accuracy"]) +
             " $\\scriptsize{$\\pm " +
             "{:.3f}".format(json_data["test_losses"]["std"]["accuracy"]) +
            "$}")
def get_test_ce(json_data):
    """
    Returns Cross-Entropy Loss on test-set averaged over all folds including standard deviation as a string
    :param json_data: JSON entry for current experiment
    :return: string for test CE-Loss in the form of '0.12 +- 0.02'
    """
    return ("$" +
            "{:.3f}".format(json_data["test_losses"]["mean"]["cross_entropy"]) +
             " $\\scriptsize{$\\pm " +
             "{:.3f}".format(json_data["test_losses"]["std"]["cross_entropy"]) +
            "$}")
def get_mae(json_data, split="train"):
    """
    Returns MAE on specified split averaged over all folds including standard deviation as a string
    :param json_data: JSON entry for current experiment
    :param split: split to calculate MAE on
    :return: MAE in the form of '0.12 +- 0.02' for selected split
    """
    return ("$" +
            "{:.3f}".format(json_data[split+"_losses"]["mean"]["mae"]) +
             " $\\scriptsize{$\\pm " +
             "{:.3f}".format(json_data[split+"_losses"]["std"]["mae"]) +
            "$}")
def get_mse(json_data, split="test"):
    """
    Returns MSE on specified split averaged over all folds including standard deviation as a string
    :param json_data: JSON entry for current experiment
    :param split: split to calculate MSE on
    :return: MSE in the form of '0.12 +- 0.02' for selected split
    """
    return ("$" +
            "{:.3f}".format(json_data[split+"_losses"]["mean"]["mse"]) +
             " $ \\scriptsize{$\\pm " +
             "{:.3f}".format(json_data[split+"_losses"]["std"]["mse"]) +
            "$}")

def create_table(results_json, column_titles_latex, column_values, filter_func=None, sort_by=None):
    """
    Creates a LaTeX table from given json and prints it to shell
    :param results_json: JSON from whole results.json
    :param column_titles_latex: list of title names of the columns in the LaTeX table
    :param column_values: List of functions that extract the wanted value from each list entry inside the results_json
    :param filter_func: Filter function to apply for specifying only certain experiments / results
    :param sort_by: sort by which columns
    """
    assert len(results_json) > 0
    # if no filter function specified, make filter always return True
    if filter_func is None:
        filter_func = lambda x: True  # filter = always True
    # build table
    rows = []
    for entry in results_json:
        # if current entry (json) matches filter_func
        if filter_func(entry):
            # extract values from json by applying column_values() lambda
            row = column_values(entry)
            assert len(column_titles_latex) == len(row)
            rows.append(row)
    rows = np.array(rows)
    # convert this table to pandas dataframe
    data_dict = dict()
    for i in range(len(column_titles_latex)):
        data_dict[column_titles_latex[i]] = rows[:, i]
    df = pd.DataFrame(data_dict)

    # now create a LaTeX table based on the dataframe
    finished_latex = ""
    preamble = "\\begin{tabular}{" + len(df.columns) * "|c" + "|" + "}\n\\hline\n"
    finished_latex += preamble
    # Create top row (column names)
    top_row = ""
    for col_name in column_titles_latex:
        top_row += col_name + " & "
    top_row = top_row[:-2] + "\\\\\\hline\n"
    finished_latex += top_row
    # potentially sort values by (multiple) column name(s)
    if sort_by is not None:
        df = df.sort_values(by=sort_by)
    table_np = df.to_numpy()
    # iterate through the rows of the table
    for row in range(table_np.shape[0]):
        # build LaTeX string for current row
        row_latex = ""
        for col in range(table_np.shape[1]):
            row_latex += str(table_np[row, col]) + " & "
        # remove last "&"
        row_latex = row_latex[:-2]
        # add linebreak
        row_latex += "\\\\"
        # add newline
        finished_latex += row_latex + "\n"
    # finish LaTeX table
    finished_latex += "\\hline\n"
    finished_latex += "\n\\end{tabular}\n"
    # print LaTeX table
    print(finished_latex)


def knot_dataset_table():
    """
    Creates a LaTeX table for the knot dataset
    """
    # column titles
    column_titles_latex = ["Model",
                           "Size",
                           "\# Params",
                           "Normalization",
                           #"\# Grids",
                           "zSiLU",
                           "Test Acc.",
                           "Test CE-Loss"]
    # corresponding (tuple of) functions to extract specific values from json entry
    column_values = lambda x: (get_model_name(x),
                               get_model_size(x),
                               get_model_params(x),
                               get_model_normalization(x),
                               #get_model_num_grids(x),
                               get_model_zsilu_type(x),
                               get_test_acc(x),
                               get_test_ce(x))
    # filter for knot dataset only (and CVKAN)
    filter_func = lambda x: x["dataset_name"].startswith("knot_") and x["model_name"] == "CVKANWrapper"
    # create LaTeX table
    create_table(results_json, column_titles_latex=column_titles_latex, column_values=column_values,
                 filter_func=filter_func, sort_by=["Model", "Size"])

def function_fitting_table():
    """
    Creates a LaTeX table for the fitting function datasets
    """
    # column titles
    column_titles_latex = [#"Dataset",
                           "Model",
                           "Size",
                           "\# Params",
                           #"\# Grids",
                           #"zSiLU",
                           "Test MSE",
                           "Test MAE",
                           #"Train MSE",
                           #"Train MAE",
                           ]
    # corresponding (tuple of) functions to extract specific values from json entry
    column_values = lambda x: (#get_dataset_name(x),
                               get_model_name(x),
                               get_model_size(x),
                               get_model_params(x),
                               #get_model_num_grids(x),
                               #get_model_zsilu_type(x),
                               get_mse(x, split="test"),
                               get_mae(x, split="test"),
                               #get_mse(x, split="train"),
                               #get_mae(x, split="train")
                               )
    # filter for physical circuit dataset with 100k samples
    filter_func = lambda x: x["dataset_name"].startswith("ph_circ") and x["dataset_name"].endswith("100k")
    # filter for general function fitting datasets (synthetic)
    #filter_func = lambda x: x["dataset_name"].startswith("ff_")
    # create LaTeX table
    create_table(results_json, column_titles_latex=column_titles_latex, column_values=column_values,
                 filter_func=filter_func, sort_by=["Model"])



if __name__ == "__main__":
    p = Path("/src/experiments/results.json")
    with open(p, "r", encoding="utf-8") as f:
        results_json = json.load(f)

    knot_dataset_table()
    #function_fitting_table()


