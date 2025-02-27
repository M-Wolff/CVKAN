"""
File: explain_kan.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Explain KAN models by calculating edge relevance scores in the same way as Ziming Liu's pyKAN 2.0
"""
import torch

from ..models.wrapper import WrapperTemplate


class KANExplainer():
    """
    Calculate relevances scores for given model using the same approach as in Ziming Liu's pyKAN 2.0.
    """

    def __init__(self, model: WrapperTemplate, samples, method="pykan"):
        """
        :param model: Model to explain
        :param samples: samples to explain the model on
        :param method: method for explaining (currently only pykan is supported)
        """
        self.model = model
        self.method = method
        # create dictionary to store edge and node relevance scores
        self.node_relevances = dict()
        self.edge_relevances = dict()
        # remember samples
        self.samples = samples
        if method == "pykan":
            self.calc_relevances_pykan()
        else:
            raise NotImplementedError("Method {} not implemented".format(method))

    def calc_relevances_pykan(self):
        """
        Calculate the relevance scores of the model by using the edge and node acitvations stored within the model
        (WrapperTemplate)
        """
        # collect activations of model's edges and nodes for the samples specified
        with torch.no_grad():
            self.model.gather_activations(self.samples.to("cuda"))
        num_layers = len(self.model.get_kan_layers())
        # iterate through the layers in reverse
        for layer_idx in reversed(range(num_layers)):
            layer = self.model.get_kan_layers()[layer_idx]
            num_neurons_in_layer = self.model.get_layersizes()[layer_idx]
            num_neurons_out_layer = self.model.get_layersizes()[layer_idx + 1]
            if layer_idx == num_layers - 1:  # last layer
                # Set Output Node Relevances to 1
                for j in range(num_neurons_out_layer):  # possibly multiple output neurons
                    self.node_relevances[(layer_idx+1, j)] = 1
            # iterate over all neurons in the previous layer
            for i in range(num_neurons_in_layer):
                outgoing_relevances = 0  # outgoing relevance of the current node i in layer (layer_idx -1)
                for j in range(num_neurons_out_layer):  # look at edge E_{layer_idx-1, i, j}
                    # Formula 9 from Kan 2.0 Paper with (probably) fixed Notation
                    a_lpj = self.node_relevances[(layer_idx + 1, j)]
                    # calculate edge and node score as they did in KAN 2.0 paper
                    e_lm_ij = torch.std(self.model.get_edge_activations((layer_idx, i, j)), dim=0)
                    n_lp_j = torch.std(self.model.get_node_activations((layer_idx + 1, j)), dim=0)
                    b_lij =  a_lpj * e_lm_ij / n_lp_j
                    # sum up all outgoing edge's relevance scores for source node's relevance score
                    outgoing_relevances += b_lij
                    # store edge relevance b_lij
                    self.edge_relevances[(layer_idx, i, j)] = b_lij
                self.node_relevances[(layer_idx, i)] = outgoing_relevances
    def get_edge_relevance(self, l_i_j_index):
        """
        Returns edge relevance score for edge connection node i in layer l with node j in the next layer
        :param l_i_j_index: tuple (l,i,j)
        :return: relevance score of edge E_{l,i,j}
        """
        return self.edge_relevances[l_i_j_index]
    def get_node_relevance(self, l_i_index):
        """
        Returns node relevance score for node i in layer l
        :param l_i_index: tuple (l, i)
        :return: relevance score of node N_{l,i}
        """
        return self.node_relevances[l_i_index]


class DummyKANExplainer(KANExplainer):
    """
    Dummy KAN Explainer that always returns 1 (placeholder if gather_activations(), ... are not implemented by the model
    """
    def __init__(self):
        pass
    def calc_relevances_pykan(self):
        pass
    def get_edge_relevance(self, l_i_j_index):
        return 1
    def get_node_relevance(self, l_i_index):
        return 1