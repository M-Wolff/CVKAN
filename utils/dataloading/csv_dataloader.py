"""
File: csv_dataloader.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Dataloader and Dataset-Class for a comma-seperated CSV file or dictionary
"""
import pathlib
from pathlib import Path
import random
import pandas
import torch
import math
from torch.utils.data import Dataset


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,input_vars,output_vars, categorical_vars, train_test="70:30", shuffle=True):
        """
        :param csv_file: file path to CSV file OR already loaded pandas dataframe OR dictionary
        :param input_vars: names of input variables (list)
        :param output_vars: names of output variables (list)
        :param categorical_vars: list of categorical variables that need to be converted to numeric values
        :param train_test: share of train and test data given as string 'Train:Test' i.e. '70:30'
        :param shuffle: flag to shuffle randomly before splitting the dataset
        """
        super().__init__()
        assert set(input_vars).isdisjoint(set(output_vars)), "Input and Output Variables cannot be overlapping sets!"
        self.input_varnames = input_vars
        self.output_varnames = output_vars
        self.categorical_vars = categorical_vars
        self.returnsplit = "train"  # The split self.__getitem__() should grab samples from.
        self.data = dict()
        print(
            f"Loading Dataset with input_vars={input_vars} and output_vars={output_vars} using categorical_vars={categorical_vars}.")
        # if csv_file is an actual file or a path
        if type(csv_file) == Path or type(csv_file) == pathlib.PosixPath:
            # load the file's contents using pandas
            print(f"Loading from Path {csv_file}")
            dataset_pandas = pandas.read_csv(csv_file)
            # restrict loaded dataset to input and output vars only
            dataset_pandas = dataset_pandas[input_vars + output_vars]
            # drop NA values
            dataset_pandas = dataset_pandas.dropna(axis=0, how="any")
        elif type(csv_file) == pandas.DataFrame:  # if csv_file is already a pandas dataset
            print("Using pre-loaded CSV Dataset passed as parameter")
            # just copy it
            dataset_pandas = csv_file
        elif type(csv_file) == dict:  # if csv_file is a dict
            # copy dict's contents to corresponding places in self.data
            self.data["train_input"] = csv_file["train_input"]
            self.data["train_label"] = csv_file["train_label"]
            self.data["test_input"] = csv_file["test_input"]
            self.data["test_label"] = csv_file["test_label"]
            return  # and return (we are done here in __init__()) !!!
        else:
            raise NotImplementedError(f"Dataset type {type(csv_file)} is not supported")
        # now dataset_pandas contains a pandas dataset that we need to convert to dictionary
        # create placeholder for train_input (for now everything ends up in train split)
        self.data["train_input"] = torch.zeros(size=(len(dataset_pandas), len(input_vars)), dtype=torch.float32)
        if len(output_vars)==1 and output_vars[0] in categorical_vars:  # Classification
            print("Amounts of samples per class: ", dataset_pandas[output_vars[0]].value_counts())
            self.tasktype = "classification"
            self.classes = list(dataset_pandas.groupby(output_vars[0])[output_vars[0]].unique())
            self.num_classes = len(self.classes)
            # create placeholder for One-Hot encoded classlabels
            self.data["train_label"] = torch.zeros(size=(len(dataset_pandas), self.num_classes), dtype=torch.float32)
            # output varnames are multiplied by num_classes because of One-Hot encoding
            self.output_varnames = self.num_classes * self.output_varnames
        else:  # Regression
            # create placeholder for output vars
            self.data["train_label"] = torch.zeros(size=(len(dataset_pandas), len(output_vars)), dtype=torch.float32)
            self.tasktype = "regression"

        # iterate over input_vars
        for i, col in enumerate(input_vars):
            if col in categorical_vars:  # if variable is categorical
                # encode categorical features as 0,1,2,...
                self.data["train_input"][:, i] = torch.from_numpy(
                    dataset_pandas[col].astype("category").cat.codes.values.copy())
                print("Converted categorical variables ", col, dataset_pandas[col].unique(), " to ",
                      torch.unique(self.data["train_input"][:, i]), " for col ", col)
            else:  # feature is numeric, so just copy the column into dict
                self.data["train_input"][:, i] = torch.from_numpy(dataset_pandas[col].values)
        # iterate over output_vars
        for i, col in enumerate(output_vars):
            if col in output_vars:
                if self.tasktype == "classification":
                    # encode categorical classnames as numbers 0,1,2,...
                    class_codes = torch.from_numpy(
                        dataset_pandas[col].astype("category").cat.codes.values.copy()).to(dtype=torch.int64)
                    # save the mapping (i.e. which class names ended up as which number)
                    class_classid_mapping = torch.unique(torch.stack([torch.from_numpy(dataset_pandas[col].to_numpy()), class_codes.squeeze()], dim=1), dim=0)
                    print("Converted categorical variables ", dataset_pandas[col].unique(), " to ",
                          torch.unique(class_codes), " for col ", col, "mapping", class_classid_mapping)
                    # convert encoded classnames into One-Hot encoded Tensors
                    self.data["train_label"][:, :] = torch.nn.functional.one_hot(class_codes, self.num_classes).float()
                else:
                    # if regression, just copy the target values into the right column
                    self.data["train_label"][:, i] = torch.from_numpy(dataset_pandas[col].values).float()

        # make sure everything is float
        self.data["train_input"] = self.data["train_input"].float()
        # calculate fraction of train and test
        train_share = int(train_test.split(":")[0])
        test_share = int(train_test.split(":")[1])
        train_fraction = train_share / (test_share + train_share)

        # Split into train and test
        indices = list(range(self.data["train_label"].size()[0]))  # all indices
        if shuffle:  # maybe randomly shuffle
            random.shuffle(indices)
        # draw test-samples out of train-split (until now everything was inside train-split)
        self.data["test_input"] = self.data["train_input"][indices[math.ceil(train_fraction * len(indices)):]]
        self.data["test_label"] = self.data["train_label"][indices[math.ceil(train_fraction * len(indices)):]]
        self.data["train_input"] = self.data["train_input"][indices[:math.ceil(train_fraction * len(indices))]]
        self.data["train_label"] = self.data["train_label"][indices[:math.ceil(train_fraction * len(indices))]]

    def to(self, device):
        """
        Move dataset to device
        :param device: device
        """
        if type(device) == str:
            device = torch.device(device)
        self.data["test_input"] = self.data["test_input"].to(device)
        self.data["test_label"] = self.data["test_label"].to(device)
        self.data["train_input"] = self.data["train_input"].to(device)
        self.data["train_label"] = self.data["train_label"].to(device)
    def get_train_test_size(self):
        """
        Get train and test size (number of samples)
        :return: Tuple (train_size, test_size)
        """
        return self.data["train_input"].size()[0], self.data["test_input"].size()[0]

    def __or__(self, other):
        """
        Unite two CSVDatasets with each other
        :param other: the other CSVDataset to unite with
        :return: joined dataset
        """
        assert type(other) == CSVDataset, "Expected other dataset to unite with to also be of class CSVDataset!"
        assert other.input_varnames == self.input_varnames, "Can only unite datasets with same input_varnames"
        assert other.output_varnames == self.output_varnames, "Can only unite datasets with same output_varnames"
        assert other.categorical_vars == self.categorical_vars, "Can only unite datasets with same categorical_vars"
        # calculate new number of samples in train and test split (add them together)
        new_train_length = self.get_train_test_size()[0] + other.get_train_test_size()[0]
        new_test_length = self.get_train_test_size()[1] + other.get_train_test_size()[1]

        print(f"Joining Dataset of size {self.get_train_test_size()} with Dataset of size {other.get_train_test_size()}...")
        print("Be careful that all categorical values have the same corresponding codes!")
        # create placeholders for the joined dataset's samples
        new_train_input =torch.zeros(size=(new_train_length, len(self.input_varnames)), dtype=torch.float32)
        new_train_label=torch.zeros(size=(new_train_length, len(self.output_varnames)), dtype=torch.float32)
        new_test_input =torch.zeros(size=(new_test_length, len(self.input_varnames)), dtype=torch.float32)
        new_test_label=torch.zeros(size=(new_test_length, len(self.output_varnames)), dtype=torch.float32)

        # copy everything from self to new dataset
        new_train_input[0:self.data["train_input"].size()[0], :] = self.data["train_input"]
        new_train_label[0:self.data["train_label"].size()[0], :] = self.data["train_label"]
        new_test_input[0:self.data["test_input"].size()[0], :] = self.data["test_input"]
        new_test_label[0:self.data["test_label"].size()[0], :] = self.data["test_label"]

        # add every sample from 'other' to new dataset
        new_train_input[self.data["train_input"].size()[0]:, :] = other.data["train_input"]
        new_train_label[self.data["train_label"].size()[0]:, :] = other.data["train_label"]
        new_test_input[self.data["test_input"].size()[0]:, :] = other.data["test_input"]
        new_test_label[self.data["test_label"].size()[0]:, :] = other.data["test_label"]

        # build dictionary
        joined_dataset = dict()
        joined_dataset["train_input"] = new_train_input
        joined_dataset["train_label"] = new_train_label
        joined_dataset["test_input"] = new_test_input
        joined_dataset["test_label"] = new_test_label
        joined_dataset = CSVDataset(joined_dataset, self.input_varnames, self.output_varnames, self.categorical_vars)
        joined_dataset.returnsplit = self.returnsplit
        if hasattr(self.num_classes):
            joined_dataset.num_classes = self.num_classes
        joined_dataset.tasktype = self.tasktype
        print(f"Joined Dataset has size {joined_dataset.get_train_test_size()} (Train, Test)")
        return joined_dataset


    def __len__(self):
        """
        Get length of dataset
        :return: length of dataset
        """
        # return length for selected split
        if self.returnsplit == "train":
            return self.data["train_input"].size()[0]
        elif self.returnsplit == "test":
            return self.data["test_input"].size()[0]
    def __getitem__(self, index):
        """
        Get sample at index 'index'
        :param index: index to get sample from
        :return: sample as a tuple (x,y)
        """
        # return sample from selected split
        if self.returnsplit == "train":
            return self.data["train_input"][index], self.data["train_label"][index]
        if self.returnsplit == "test":
            return self.data["test_input"][index], self.data["test_label"][index]
    def set_returnsplit(self, return_split):
        """
        Update the dataset's split to return when used in a dataloader
        :param return_split: new returnsplit. Either 'train' or 'test'
        """
        self.returnsplit = return_split
    def normalize(self):
        """Scale and shift each feature seperately so that all values of that feature fall in range  [-2, 2]"""
        # iterate over all features
        for i, feature_name in enumerate(self.input_varnames):
            # collect mins and maxs across both train and test split, if each not empty
            mins = []
            maxs = []
            # Train and Test set might be empty
            if self.get_train_test_size()[1] > 0:
                mins.append(torch.amin(self.data["test_input"][:,i]))
                maxs.append(torch.amax(self.data["test_input"][:, i]))
            if self.get_train_test_size()[0] > 0:
                mins.append(torch.amin(self.data["train_input"][:,i]))
                maxs.append(torch.amax(self.data["train_input"][:, i]))
            min_ = torch.min(torch.Tensor(mins))
            max_ = torch.max(torch.Tensor(maxs))
            # scale so that current feature covers interval [-2, 2] completely (i.e. shift so that new min is zero,
            # then scale to [0, 1] and then re-scale to [-2, 2]
            if self.get_train_test_size()[1] > 0:
                self.data["test_input"][:, i] = (self.data["test_input"][:, i] - min_) / (max_ - min_) * 4 - 2
            if self.get_train_test_size()[0] > 0:
                self.data["train_input"][:,i] = (self.data["train_input"][:,i] - min_) / (max_ - min_) * 4 - 2
    def normalize_for_pykan(self):
        """
        Normalize dataset for pyKAN, which expects dataset to have std of 1 and mean of zero.
        """
        mean_train = torch.mean(self.data["train_input"], dim=0)
        std_train = torch.std(self.data["train_input"], dim=0)

        self.data["train_input"] = (self.data["train_input"] - mean_train) / std_train
        if self.get_train_test_size()[1] > 0:
            self.data["test_input"] = (self.data["test_input"] - mean_train) / std_train