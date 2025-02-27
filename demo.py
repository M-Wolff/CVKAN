from cvkan import CVKANWrapper, train_kans, KANPlot
from cvkan.models.CVKAN import Norms
from cvkan.utils import create_complex_dataset, CSVDataset
from cvkan.utils.loss_functions import MSE, MAE


# Generate dataset for f(z)=(z1^2 + z2^2)^2
f_squaresquare = lambda x: (x[:,0]**2 + x[:,1]**2)**2
# create dataset (this is a dictionary with keys 'train_input', 'train_label', 'test_input' and 'test_label', each
# containing a Tensor as value)
dataset = create_complex_dataset(f=f_squaresquare, n_var=2, ranges=[-1,1], train_num=5000, test_num=1000)
# convert dataset to CSVDataset object for easier handling later
dataset = CSVDataset(dataset, input_vars=["z1", "z2"], output_vars=["(z1^2 + z2^2)^2"], categorical_vars=[])


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

# plot the model
kan_plotter = KANPlot(cvkan_model,
                      kan_explainer=None,
                      input_featurenames=dataset.input_varnames,
                      output_names=dataset.output_varnames,
                      complex_valued=True,
                      )
kan_plotter.plot_all()