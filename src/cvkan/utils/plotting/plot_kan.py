"""
File: plot_kan.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Plot KAN models (real-valued as well as complex-valued) with interactive elements.
Features:
- Transparency can be adjusted with a slider (with respect to the calculated relevance scores by KANExplainer)
- double-click on nodes only shows incoming and outgoing edge of that specific node. Can be undone by double-clicking
  anywhere else inside the plot
- the function of one single edge can be visualized as a close-up detailed version on the right. Selection is done by
  entering the tuple of the edge (l,i,j) for edge E_{l,i,j} in the text-box and pressing enter.
"""
import math
import matplotlib.colors
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, TextBox

from ...models.wrapper import WrapperTemplate
from ..explain_kan import KANExplainer, DummyKANExplainer
from . import cplotting_tools as cplot

# Default Values for Plot sizes
_NODE_PLOT_CIRCLESIZE = 0.05  # Node circle size
_ACTPLOT_SIZE = (0.7,0.7)  # size of plotted activation functions within network plot (rectangular sub-plot)
_PLOT_AT_LINK_FRACTION = 0.5  # Plot activation function at this fraction of a connection's link (0.5 = at middle of vertex)
_COLORS = list(matplotlib.colors.BASE_COLORS)  # colormap for nodes and their outgoing edges within each layer
_COLORS.remove("w")  # plotting in white is pointless
_LINESTYLES = ["solid", "dashed", "solid"]  # for base, splines, sum (real-valued only)

matplotlib.use('TkAgg')  # use a Backend that supports interactive elements

class KANPlot():
    def __init__(self, model: WrapperTemplate, kan_explainer: KANExplainer=None, input_featurenames=None, output_names=None, plot_options=None, complex_valued=False):
        """
        :param model: Model to Plot (should implement the methods of WrapperTemplate)
        :param kan_explainer: For relevance-scores per edge (set to None to disable KAN explainer)
        :param input_featurenames: Input Featurenames displayed in the plot
        :param output_names: Output Names displayed in the plot
        :param plot_options: Boolean Array for default visibility of components (just leave at None and manually play around with the Checkboxes in the plot)
        :param complex_valued: Switch to plotting a Complex-Valued KAN
        """
        self.model = model
        self.model.to(torch.device("cpu"))
        if kan_explainer is None:
            # if no explainer is given, use a dummy explainer that always outputs node's and edge's relevance as 1
            self.kan_explainer = DummyKANExplainer()
        else:
            self.kan_explainer = kan_explainer
        self.input_featurenames = input_featurenames
        self.output_names = output_names
        self.complex_valued = complex_valued
        # scale up x-axis to make the plot less elongated
        self.xscale_factor = len(self.model.get_kan_layers()) + 2
        # +0.5 for centering, * _XSCALE_FACTOR to make plot wider and square-ish (otherwise plot would be super elongated and hard to read)
        # using same axis scaling alone is not enough since that makes circles ellipses and stretches activation function plots
        self.nodeid2xpos = lambda k, dimsize: (k + 0.5) / dimsize * self.xscale_factor  # node id within layer to xpos
        self.xpos2nodeid = lambda x, dimsize: (x / self.xscale_factor * dimsize) - 0.5  # xpos to node id within one layer
        # for converting relevance score to transparency
        self.beta = 0.3
        # dictionary containing the plotted activation functions with key [function_type][(l,i,j)]
        self.act_functs = dict()
        # and their corresponding coordinate axes
        self.act_functs_coordinatesystems = dict()
        # links between nodes with keys (l,i,j)
        self.connections = dict()
        # textboxes for input featurenames with keys (i)
        self.textboxes_input_featurenames = dict()
        self.edge_relevances = dict()  # key (l,i,j) tuple
        self.node_relevances = dict()  # key: (l,i) tuple
        # nodes with key (l,i)
        self.nodes = dict()
        # Dict for storing images (screenshots) in complex valued mode or potentially lists of matplotlib Line objects
        self.complex_images = dict()
        self.complex_draw_colorbar = True  # 'counter' to plot colorbar only once
        self.plot_options = plot_options
        self.lij_last_choice = (0,0,0)  # function to plot in detail view on the right (remember choice for updates)
        if self.plot_options is None:
            # plot options exist for base, splines, act_funct and coordinate_axes
            # base is i.e. SiLu, splines are the individual functions around grid-points (i.e. splines or RBFs),
            # act_funct is the sum of everything and coordinate_axes are 2D coordinate axes to be plotted (only real-valued)
            self.plot_options = [False, False, True, True]
        self.function_types = ["base", "splines", "act_funct", "coordinate_axes"]
        for ft in self.function_types:
            self.act_functs[ft] = dict()
        # remember which visibilities have been updated to revert back to them
        self.visibilities_to_revert_to = []
        # which node is currently in focus (double-clicked)
        self.currently_focused_node = None

        if self.complex_valued:
            # if complex-valued the right detail plot needs to be a 3d projection
            fig = plt.figure(figsize=plt.figaspect(2.))
            axs = [None, None]
            axs[0] = fig.add_subplot(1, 2, 1)
            axs[1] = fig.add_subplot(1, 2, 2, projection='3d')
        else:
            # for real-valued just create 2 subplots
            fig, axs = plt.subplots(figsize=(8, 8), ncols=2, nrows=1, width_ratios=[2,1])
        axs[0].axis("on")
        axs[0].set_aspect("equal")
        self.fig = fig
        self.axs = axs

        if not self.complex_valued:
            # Normalize the functions per Layer to make the plot more readable
            # it might also be better to turn this off sometimes (just append 1.0 instead of max(abs(act_ys))
            self.max_activation_value = dict()  # this is supposed to store the max activation value per layer
            for layer_id, layer in enumerate(model.get_kan_layers()):
                # create empty list for current layer
                self.max_activation_value[layer_id] = []
                for i in range(self.model.get_layersizes()[layer_id]):
                    for j in range(self.model.get_layersizes()[layer_id+1]):
                        act_xs, act_ys = self.model.plot_curve((layer_id,i,j))
                        act_ys = act_ys[2]  # only care about sum of splines and silu (not the individual components themselves)
                        # append max of each edge's activation function within this layer
                        self.max_activation_value[layer_id].append(max(abs(act_ys)))
            for k in self.max_activation_value.keys():  # take max within each layer k
                self.max_activation_value[k] = np.max(self.max_activation_value[k])
        # Create Interactive Sliders and Checkboxes
        slider_ax = plt.axes([0.05, 0.2, 0.05, 0.6])
        # Sliders and Checkboxes ***MUST*** be stored as attributes of the class
        # otherwise stuff breaks (maybe the Garbage Collector deletes them, although they should be used later on?)...
        # took me ~6 hours to find this out. Amazing.
        self.slider = Slider(slider_ax, label="beta", valmin=0, valmax=2, valinit=0.5, orientation="vertical")
        # if slider value changes, call self.update_transparency() with new slider value as parameter
        self.slider.on_changed(self.update_transparency)
        # checkboxes for switching on/off the individual function types. Only implemented for real-valued functions
        if not self.complex_valued:
            cb_ax = plt.axes([0.2, 0.8, 0.2, 0.2])
            self.cb_buttons = CheckButtons(ax=cb_ax, labels=self.function_types, actives=self.plot_options)
            self.cb_buttons.on_clicked(self.update_plot_options)
    def plot_all(self):
        """
        Plot the entire model together with detailed view on the right
        """
        # plot model structure
        self.plot_model()
        # and one single activation function in detail
        self.plot_single_function()
        plt.show()

    def calc_opaqueness(self, relevance):
        """
        Calculate the opaqueness of an Edge / Node based on it's relevance score
        """
        result = self.beta * math.tanh(relevance)  # same as for pyKAN 2.0
        return np.clip(result, 0, 1)  # clip to range [0,1]
    def update_transparency(self, slider_value=None):
        """
        Updates the transparency of Edges / Nodes, whenever the Slider is changed
        """
        self.undo_visibility_focus()
        if slider_value is not None:
            self.beta = slider_value
        # iterate through all the visible components in the plot and change their alpha values
        # activation functions (each function_type)
        for function_type in self.act_functs.keys():
            for l_i_j_index in self.act_functs[function_type].keys():
                for spline_function in self.act_functs[function_type][l_i_j_index]:
                    spline_function.set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(l_i_j_index)))
                    # update their visibilities as desired (invisible / visible) as self.plot_options dictates
                    if not self.plot_options[self.function_types.index(function_type)]:
                        spline_function.set_visible(False)
                    else:
                        spline_function.set_visible(True)
                # set alpha of vertices
                self.connections[l_i_j_index].set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(l_i_j_index)))
        # also update the plotted coordinatesystems
        for k in self.act_functs_coordinatesystems.keys():
            if not self.plot_options[3]:
                for gridline in self.act_functs_coordinatesystems[k]:
                    gridline.set_visible(False)
            else:
                for gridline in self.act_functs_coordinatesystems[k]:
                    gridline.set_visible(True)
                    gridline.set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(k)))
        # update nodes
        for k in self.nodes.keys():
            self.nodes[k].set_alpha(self.calc_opaqueness(self.kan_explainer.get_node_relevance(k)))
        # update textboxes (disabled for now)
        for k in self.textboxes_input_featurenames.keys():
            pass
            #self.textboxes_input_featurenames[k].set_alpha(self.calc_opaqueness(self.kan_explainer.get_node_relevance((0,k))))
        # update screenshots of complex-valued functions (or the lineset objects)
        for k in self.complex_images.keys():
            if type(self.complex_images[k]) == list:  # list of lineset objects
                for el in self.complex_images[k]:
                    el.set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(k)))
            else:  # images (single item)
                self.complex_images[k].set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(k)))
            self.connections[k].set_alpha(self.calc_opaqueness(self.kan_explainer.get_edge_relevance(k)))
        self.fig.canvas.draw_idle()
        self.focus_on_node()  # keep focused node still focused
    def update_plot_options(self, label):
        """This method is called whenever the Checkboxes are touched. Update the visibility of the Activation Function's
        components according to the selection
        :param label: new plot options (id of the checkbox changed)
        """
        # flip plot_option for this function_type
        self.plot_options[self.function_types.index(label)] = not self.plot_options[self.function_types.index(label)]
        # update the transparencies and visibilities
        self.update_transparency()
        # also update the single Activation Function plot (detail view) based on the Checkbox selection
        self.update_single_function(l_i_j_index=self.lij_last_choice)
    def plot_model(self):
        """
        Plot the Model structure (left)
        """
        # plot all layers
        for i in range(len(self.model.get_kan_layers())):
            current_layer = self.model.get_kan_layers()[i]
            self.plot_layer(layer_id=i)

        # plot feature names
        if self.input_featurenames is not None:
            # add linebreaks in very long featurenames...
            input_featurenames = [x.split(".")[-1] for x in self.input_featurenames]
            for i in range(len(input_featurenames)):
                linebroken = ""
                for j, c in enumerate(input_featurenames[i]):
                    linebroken += c
                    if j % 10 == 0 and j > 0:
                        linebroken += "\n"
                input_featurenames[i] = linebroken
            assert self.model.get_layersizes()[0] == len(input_featurenames)
            # plot input featurenames
            for i in range(self.model.get_layersizes()[0]):
                textbox_xpos = self.nodeid2xpos(i, self.model.get_layersizes()[0])
                self.textboxes_input_featurenames[i] = self.axs[0].text(textbox_xpos, -0.2, input_featurenames[i], horizontalalignment='center',
                         verticalalignment='center', rotation=90, fontsize=16, bbox=dict(facecolor='none', edgecolor='red'))
        # plot output names
        for i in range(self.model.get_layersizes()[-1]):
            if self.output_names is not None:
                assert self.model.get_layersizes()[-1] == len(self.output_names)
                textbox_xpos = self.nodeid2xpos(i, self.model.get_layersizes()[-1])
                self.axs[0].text(textbox_xpos, len(self.model.get_kan_layers()) + 0.25, self.output_names[i], horizontalalignment='center',
                                 verticalalignment='center')
            # plot output node(s)
            output_node_circle = plt.Circle((self.nodeid2xpos(i, self.model.get_layersizes()[-1]), len(self.model.get_layersizes()) - 1),
                                            radius=_NODE_PLOT_CIRCLESIZE,
                                            color="black")
            self.nodes[len(self.model.get_layersizes()) - 1, i] = output_node_circle
            self.axs[0].add_patch(output_node_circle)

        # set ylim to not cut away feature names or output names
        self.axs[0].set_ylim(-0.5, len(self.model.get_kan_layers()) + 0.5)
        # clicks (later checked if it was double-click) lead to focus on single node
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.update_transparency()
    def focus_on_node(self):
        """
        Only display incoming and outgoing edges of self.currently_focused_node
        (which is set at another place in the code).
        """
        # revert changes for focusing on previous node (might already be focusing on some node)
        self.undo_visibility_focus()
        if self.currently_focused_node is None:  # no new node to focus on
            return
        # node to focus on
        layer_id, node_id = self.currently_focused_node
        # Turn every edge that's not incoming or outgoing of Node (layer_id, node_id) invisible
        for (l, i), node in self.nodes.items():
            if i == node_id and l == layer_id:  # node within current layer should stay visible
                continue
            if layer_id == l - 1 or layer_id == l + 1:  # Nodes in the previous and next Layer should also stay visible
                continue
            # remember the changes done so we can go back to the original plot later on
            self.visibilities_to_revert_to.append((node, node.get_visible()))
            node.set_visible(False)
        # do the same for activation functions
        for ft in self.function_types:
            for (l, i, j), act_funs in self.act_functs[ft].items():
                if (l == layer_id and i == node_id) or (l == layer_id - 1 and j == node_id):
                    continue
                for act_fun in act_funs:
                    self.visibilities_to_revert_to.append((act_fun, act_fun.get_visible()))
                    act_fun.set_visible(False)
        # and edges
        for (l, i, j), edge in self.connections.items():
            if (l == layer_id and i == node_id) or (l == layer_id - 1 and j == node_id):
                continue
            self.visibilities_to_revert_to.append((edge, edge.get_visible()))
            edge.set_visible(False)
            if (l,i,j) in self.complex_images:
                lines = self.complex_images[(l,i,j)]
                if type(lines) != list:
                    lines = [lines]
                for line in lines:
                    self.visibilities_to_revert_to.append((line, line.get_visible()))
                    line.set_visible(False)
        # and the coordinate systems plotted inside the model's structure
        for (l, i, j), act_fun_coord_systems in self.act_functs_coordinatesystems.items():
            if (l == layer_id and i == node_id) or (l == layer_id - 1 and j == node_id):
                continue
            for act_fun_coord_sys in act_fun_coord_systems:
                self.visibilities_to_revert_to.append((act_fun_coord_sys, act_fun_coord_sys.get_visible()))
                act_fun_coord_sys.set_visible(False)

        self.fig.canvas.draw_idle()
    def on_mouse_press(self, event):
        """This function is called whenever a mouse-click happens. It tries to match the double click's position to a node
        and then makes every Edge / Node, that is not directly connected to it, invisible.
        :param event: mouse click event
        """
        if not event.dblclick:  # Has to be double click
            return
        # wherever the click happens: undo the last focus on node
        self.undo_visibility_focus()
        if event.xdata is None or event.ydata is None:
            return
        self.currently_focused_node = None
        # Try to convert event.ydata to layer index
        # double-click has to happen within node circle vertically (so we first match the layer)
        if abs(round(event.ydata) - event.ydata) <= _NODE_PLOT_CIRCLESIZE:
            layer_id = round(event.ydata)
        else:
            print("failed to match layer")
            return
        # Try to find matching node within this layer
        node_candidate_id = self.xpos2nodeid(event.xdata, self.model.get_layersizes()[layer_id])
        node_candidate_id = int(node_candidate_id)
        node_candidate_xpos = self.nodeid2xpos(node_candidate_id, self.model.get_layersizes()[layer_id])
        # if nearest node is close enough to mouse pointer (horizontally)
        if abs(node_candidate_xpos - event.xdata) <= _NODE_PLOT_CIRCLESIZE:
            node_id = round(node_candidate_id)
        else:
            print("failed to match node")
            return
        # set node to focus on
        self.currently_focused_node = (layer_id, node_id)
        self.focus_on_node()  # actually do the visibility updates
    def undo_visibility_focus(self):
        # Undo all the visibility changes made by on_mouse_press
        for line, state in self.visibilities_to_revert_to:
            line.set_visible(state)
        self.visibilities_to_revert_to = []
        self.fig.canvas.draw_idle()
    def plot_act_funct(self, x_center,y_center,data_x,functions_y, l_i_j_index, color="black"):
        """
        Plot a single Activation Function inside the Model's structure plot
        :param x_center: plot around xcenter
        :param y_center: plot around ycenter
        :param data_x: x-positions of the activation function
        :param functions_y: y-values of the activation function as an array [base-function, splines, resulting act function]
        :param l_i_j_index: layer,i,j index of the corresponding Edge
        :param color: color to plot the function in
        """
        convert_x_to_xpos_global = lambda x: _ACTPLOT_SIZE[0] * x + x_center
        convert_y_to_ypos_global = lambda y: y / torch.amax(torch.abs(y)) * _ACTPLOT_SIZE[1] + y_center
        # convert_y_to_ypos_global = lambda y: y / self.max_activation_value[l_i_j_index[0]] * _ACTPLOT_SIZE[1] + y_center

        if self.complex_valued:  # complex-valued functions are treated differently
            rect = [x_center-_ACTPLOT_SIZE[0], y_center-_ACTPLOT_SIZE[0],_ACTPLOT_SIZE[0], _ACTPLOT_SIZE[1]]
            rect = [x_center - _ACTPLOT_SIZE[0], x_center + _ACTPLOT_SIZE[0], y_center - _ACTPLOT_SIZE[1], y_center + _ACTPLOT_SIZE[1]]
            # get the function's values
            (xs, ys), (basis, rbfs, zs) = self.model.plot_curve(l_i_j_index, num_pts=100)
            # plot the complex-valued function and return as image / screenshot
            img = cplot.complex_plot3D(xs, ys, zs, draw_colorbar=False, return_image=True, fontsize=32)
            # embed the screenshot into the current plot
            displayed_img = self.axs[0].imshow(img, extent=rect)
            # and store it in self.complex_images for later transparency updates
            self.complex_images[l_i_j_index] = displayed_img
        else:  # real-valued
            self.act_functs_coordinatesystems[l_i_j_index] = []
            for ft in self.function_types:
                self.act_functs[ft][l_i_j_index] = []
            # create a coordinate system (x- and y-axis) for this specific activation function
            yaxis = self.axs[0].plot([x_center, x_center], [-_ACTPLOT_SIZE[1] + y_center, _ACTPLOT_SIZE[1] + y_center],
                                          color="black", linewidth=0.5)[0]
            xaxis_left = convert_x_to_xpos_global(-0.5)
            xaxis_right = convert_x_to_xpos_global(0.5)
            xaxis = self.axs[0].plot([xaxis_left, xaxis_right], [y_center, y_center],
                                          color="black", linewidth=0.5)[0]
            self.act_functs_coordinatesystems[l_i_j_index].append(yaxis)
            self.act_functs_coordinatesystems[l_i_j_index].append(xaxis)

            # plot all available activation functions
            # this can be [base-function (i.e. SiLU), splines / learnable functions, sum of all of them]
            for data_y_index, data_y in enumerate(functions_y):
                if data_y is None:
                    continue
                # make data fit into the plotted coordinate system (scale and shift)
                data_x_shifted = convert_x_to_xpos_global(data_x)
                data_y_shifted = convert_y_to_ypos_global(data_y)
                # one of [base-function, learnable function, resulting act function]
                curr_func_type = self.function_types[data_y_index]
                # if learnable function, data_y is an array containing all the individual learned functions
                # (centered around different grid-points)
                if data_y_index == 1:
                    for i in range(data_y_shifted.shape[1]):
                        self.act_functs[curr_func_type][l_i_j_index].append(
                            self.axs[0].plot(data_x_shifted, data_y_shifted[:, i], color=color,
                                                  linewidth=0.5 * (data_y_index + 1), linestyle=_LINESTYLES[data_y_index])[0])
                else:  # otherwise just plot single function

                    self.act_functs[curr_func_type][l_i_j_index].append(
                        self.axs[0].plot(data_x_shifted, data_y_shifted, color=color,
                                              linewidth=0.5 * (data_y_index + 1), linestyle=_LINESTYLES[data_y_index])[0])



    def plot_layer(self, layer_id):
        """
        Plot one layer of the model
        :param layer_id: ID of the layer
        """
        # incoming and outgoing number of nodes
        in_d = self.model.get_layersizes()[layer_id]
        out_d = self.model.get_layersizes()[layer_id+1]
        # plot every Edge between this one and the next layer
        for i in range(0, in_d):
            # calculate node coordinates
            node_i_xpos = self.nodeid2xpos(i, in_d)
            for j in range(0, out_d):
                # calculate node coordinates
                node_j_xpos = self.nodeid2xpos(j, out_d)
                # get activation function values from the model
                act_xs, act_ys = self.model.plot_curve((layer_id,i,j))
                # calculate position for plotting the activation function at
                link_length = math.sqrt(1 + (node_i_xpos - node_j_xpos)**2)  # 1² + (x_i - x_j)² = hypothenuse
                plot_at_link_length = _PLOT_AT_LINK_FRACTION*link_length
                link_angle = math.asin(1/link_length)
                plot_at_ypos = math.sin(link_angle) * plot_at_link_length + layer_id
                factor = -1 if node_j_xpos < node_i_xpos else 1
                plot_at_xpos = factor * math.cos(link_angle) * plot_at_link_length + node_i_xpos
                # plot activation function using new axis
                self.plot_act_funct(plot_at_xpos, plot_at_ypos, act_xs, act_ys, (layer_id, i, j), color=_COLORS[i%len(_COLORS)])
                # plot connecting line between the nodes in subsequent functions
                self.connections[(layer_id,i,j)] = self.axs[0].plot([node_i_xpos, node_j_xpos],
                                                                         [layer_id + _NODE_PLOT_CIRCLESIZE, layer_id + 1 - _NODE_PLOT_CIRCLESIZE], color="black",
                                                                         linewidth=1)[0]

            # Draw circle for nodes in current input layer
            node_i_circle = plt.Circle((node_i_xpos, layer_id), radius=_NODE_PLOT_CIRCLESIZE, color=_COLORS[i%len(_COLORS)])
            self.nodes[(layer_id,i)] = node_i_circle
            self.axs[0].add_patch(node_i_circle)
    def plot_single_function(self):
        """
        Plot one single activation function in detail (right window part)
        """
        lij_chooser_ax = self.fig.add_axes([0.55, 0.9, 0.1, 0.1])
        # text-box to select which function should be plotted in detail
        self.lij_chooser = TextBox(lij_chooser_ax, 'l,i,j', initial="0,0,0")
        self.lij_chooser.on_submit(self.update_single_function)
        self.update_single_function(self.lij_chooser.text)
    def update_single_function(self, l_i_j_index):
        """
        Update the window with the one single activation function in detail.
        :param l_i_j_index: layer, i and j for the selected Activation function to plot as tuple (layer, i, j)
        """
        l, i, j = [int(idx) for idx in l_i_j_index.split(",")]
        self.lij_last_choice = l_i_j_index  # remember choice
        self.axs[1].cla()  # clear axis

        if self.complex_valued:  # complex-valued
            (xs, ys), (basis, rbfs, zs) = self.model.plot_curve((l,i,j))
            # plot directly into the axis of the right subfigure
            cplot.complex_plot3D(xs, ys, zs, fig_ax=(self.fig, self.axs[1]), draw_colorbar=self.complex_draw_colorbar)
            # draw the colorbar only once (otherwise they stack over each other weirdly...)
            if self.complex_draw_colorbar:
                self.complex_draw_colorbar = False

        else:  # real-valued
            # colors for base, learnable functions, resulting act function
            colors = ["black","green","blue"]
            try:
                xs, ys = self.model.plot_curve((l,i,j))
                # plot options has boolean for every function_type ["base", "splines", "act_funct", "coordinate_axes"]
                for i in range(len(self.plot_options)-1):
                    if self.plot_options[i]:  # if current function_type should be plotted
                        self.axs[1].plot(xs, ys[i], linestyle=_LINESTYLES[i], linewidth=i+1, color=colors[i])
                # create coordinate axis at x=0 and y=0
                self.axs[1].axhline(y=0, color='k', linewidth=0.5)
                self.axs[1].axvline(x=0, color='k', linewidth=0.5)
                self.fig.canvas.draw_idle()
            except ValueError:
                print("Wrong choice for l,i,j input! (needs to be 3 integers separated by comma). Given:", l_i_j_index)
            except IndexError:
                print("Wrong choice for l,i,j input! (Edge l,i,j need to exist). Given:", l_i_j_index)