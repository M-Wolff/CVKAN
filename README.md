# [CVKAN: Complex-Valued Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.02417)
Authors: Matthias Wolff, Florian Eilers, Xiaoyi Jiang \
University of Münster, Department of Computer Science

Link to Paper: https://arxiv.org/abs/2502.02417

---

### Abstract
In this work we propose $\mathbb{C}$ KAN, a complex-valued KAN, to join the intrinsic interpretability of KANs and the advantages of Complex-Valued Neural Networks (CVNNs). We show how to transfer a KAN and the necessary associated mechanisms into the complex domain. To confirm that $\mathbb{C}$ KAN meets expectations we conduct experiments on symbolic complex-valued function fitting and physically meaningful formulae as well as on a more realistic dataset from knot theory. Our proposed $\mathbb{C}$ KAN is more stable and performs on par or better than real-valued KANs while requiring less parameters and a shallower network architecture, making it more explainable.
<p align="center">
<img src="src/images/cvkan_sqsq_plot_withcolorbar.png" alt="<CVKAN Plot>" width="400"/>
</p>

---

## Table of Contents

- [src/cvkan/experiments](src/cvkan/experiments): Scripts for our experiments and corresponding results
  - [fit_formulas.py](src/cvkan/experiments/fit_formulas.py): Experiments for function fitting. Simple arbitrary $\left(z^2, \quad \sin(z), \quad z_1 \cdot z_2, \quad (z_1^2 + z_2^2)^2 \right)\quad$ as well as physically meaningful formulae (circuit & holography)
  - [knot_dataset.py](src/cvkan/experiments/knot_dataset.py): Experiments for knot classification
  - [results.json](src/cvkan/experiments/results.json): All of our results as a list of dictionaries, stored as JSON
  - [run_crossval.py](src/cvkan/experiments/run_crossval.py): Script to run k-fold cross-validation on a dataset and model given. Also stores the results with additional meta-data in a json file
- [src/images](src/images): The images used in our paper
  - [visualizations.py](src/images/visualizations.py): Script to create some of the images we used in our paper
- [src/cvkan/models](src/cvkan/models):
  - [functions](src/cvkan/models/functions): different helper functions ($\mathbb{C}$ SiLU, BatchNorms)
    - [CompleySilu.py](src/cvkan/models/functions/ComplexSilu.py): Two different variants of complex SiLU
    - [CV_LayerNorm.py](src/cvkan/models/functions/CV_LayerNorm.py): Different complex-valued BatchNorm approaches and LayerNorm
  - [wrapper](src/cvkan/models/wrapper): Folder contains Wrappers for every KAN to make them work with our KanPlotter and KanExplainer
    - [CVKANWrapper.py](src/cvkan/models/wrapper/CVKANWrapper.py): Wrapper for our CVKAN
    - [PyKANWrapper.py](src/cvkan/models/wrapper/PyKANWrapper.py): Wrapper for pyKAN
    - [WrapperTemplate.py](src/cvkan/models/wrapper/WrapperTemplate.py): Template (Interface) for all specific KAN Wrappers
  - [CVKAN.py](src/cvkan/models/CVKAN.py): $\mathbb{C}$ KAN model definition
  - [FastKAN.py](src/cvkan/models/FastKAN.py): modified version of FastKAN model definition, originally from Github Repository [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py)
- [src/cvkan/train/train_loop.py](src/cvkan/train/train_loop.py): Main loop for training all kinds of KANs on different datasets using custom loss functions
- [src/cvkan/utils](src/cvkan/utils): miscellaneous utils
  - [dataloading](src/cvkan/utils/dataloading): utils for dataloading
    - [create_complex_dataset.py](src/cvkan/utils/dataloading/create_complex_dataset.py): Create a complex-valued dataset dictionary based on a lambda expression as symbolic formula.
    - [crossval_splitter.py](src/cvkan/utils/dataloading/crossval_splitter.py): Automatically create datasets for k-fold cross-validation
    - [csv_dataloader.py](src/cvkan/utils/dataloading/csv_dataloader.py): Dataloader and Dataset-Class for a comma-seperated CSV file or dictionary
  - [latex](src/cvkan/utils/latex): Utils to generate LaTeX outputs automatically
    - [latex_table_creator.py](src/cvkan/utils/latex/latex_table_creator.py): Automatically generate resulting LaTeX tables from results.json
  - [plotting](src/cvkan/utils/plotting): utils for plotting
    - [cplot.py](src/cvkan/utils/plotting/cplot.py): Experiments with plotting standard complex-valued functions (i.e. $z^2$)
    - [cplotting_tools.py](src/cvkan/utils/plotting/cplotting_tools.py): modified version of FastKAN model definition, originally from Github Repository [artmenlope/complex-plotting-tools](https://github.com/artmenlope/complex-plotting-tools/blob/master/cplotting_tools.py)
    - [plot_kan.py](src/cvkan/utils/plotting/plot_kan.py): Plot KAN models (real-valued as well as complex-valued) with interactive elements
  - [eval_model.py](src/cvkan/utils/eval_model.py): Evaluation of models and plotting of confusion matrix
  - [explain_kan.py](src/cvkan/utils/explain_kan.py): Explain KAN models by calculating edge relevance scores in the same way as Ziming Liu's pyKAN 2.0
  - [json_editor.py](src/cvkan/utils/json_editor.py): Manipulate the results.json file
  - [loss_functions.py](src/cvkan/utils/loss_functions.py): MSE, MAE and cross entropy loss-functions
  - [misc.py](src/cvkan/utils/misc.py): Miscellaneous short scripts and methods


---

## How to use
See [demo.py](demo.py)
### Install
```bash
pip install cvkan
```
### Imports
```python
from cvkan import CVKANWrapper, train_kans, KANPlot
from cvkan.models.CVKAN import Norms
from cvkan.utils import create_complex_dataset, CSVDataset
from cvkan.utils.loss_functions import MSE, MAE
```
### Create Dataset
```python
# Generate dataset for f(z)=(z1^2 + z2^2)^2
f_squaresquare = lambda x: (x[:,0]**2 + x[:,1]**2)**2
# create dataset (this is a dictionary with keys 'train_input', 'train_label', 'test_input' and 'test_label', each
# containing a Tensor as value)
dataset = create_complex_dataset(f=f_squaresquare, n_var=2, ranges=[-1,1], train_num=5000, test_num=1000)
# convert dataset to CSVDataset object for easier handling later
dataset = CSVDataset(dataset, input_vars=["z1", "z2"], output_vars=["(z1^2 + z2^2)^2"], categorical_vars=[])
```

### $\mathbb{C}$ KAN
````python
# create CVKAN model. Note that this is CVKANWrapper, which is basically the same as CVKAN but with additional
# features for plotting later on
cvkan_model = CVKANWrapper(layers_hidden=[2,1,1], num_grids=8, use_norm=Norms.BatchNorm, grid_mins=-2, grid_maxs=2, csilu_type="complex_weight")



# train cvkan_model on dataset
results = train_kans(cvkan_model,  # model
           dataset,  # dataset
           loss_fn_backprop=MSE(),  # loss function to use for backpropagation
           loss_fns={"mse": MSE(), "mae": MAE()},  # loss function dictionary to evaluate the model on
           epochs=500,  # epochs to train for
           batch_size=1000,  # batch size for training
           kan_explainer=None,  # we could specify an explainer to make edge's transparency represent edge's relevance
           add_softmax_lastlayer=False,  # we don't need softmax after last layer (as we are doing regression)
           last_layer_output_real=False  # last layer should also have complex-valued output (regression)
           )
print("results of training: \n", results)
````
### Plotting
```python
# plot the model
kan_plotter = KANPlot(cvkan_model,
                      kan_explainer=None,
                      input_featurenames=dataset.input_varnames,
                      output_names=dataset.output_varnames,
                      complex_valued=True,
                      )
kan_plotter.plot_all()
```
In rare occasions the random initialization of the weights is suboptimal, which leads to the model not training correctly. If you do not end up with an image similar to [the one displayed above](/src/images/cvkan_sqsq_plot_withcolorbar.png) or if the resulting Test MSE is way worse than 0.05, please run again.