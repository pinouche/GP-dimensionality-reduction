from sklearn import datasets
import numpy as np
import pandas as pd


# dermatology dataset
def load_dermatology_dataset():
    path = "UCI_datasets/dermatology/dermatology.data"

    data = np.array(pd.read_csv(path, header=None, delimiter=","))
    data = np.delete(data, -2, 1)

    data_x_original, data_y_original = data[:, :-1].astype(int), data[:, -1].astype(int)
    data_y_original = data_y_original - 1

    return data_x_original, data_y_original


# dermatology dataset
def load_ionosphere_dataset():
    path = "UCI_datasets/ionosphere/ionosphere.data"

    data = np.array(pd.read_csv(path, header=None, delimiter=","))

    data_y, data_x = data[:, 0].astype(int), data[:, 1:]
    data_x = np.delete(data_x, -1, 1)

    return data_x, data_y


# wine dataset
def load_wine_dataset():
    data_dic = datasets.load_wine()
    data_x_original, data_y_original = data_dic["data"], data_dic["target"]

    return data_x_original, data_y_original


# libras dataset
def load_libras_dataset():
    path = "UCI_datasets/libras/movement_libras.data"

    data = np.array(pd.read_csv(path, header=None, delimiter=","))

    data_y, data_x = data[:, -1].astype(int), data[:, :-1]
    data_y = data_y - 1

    return data_x, data_y


# segmentation dataset
def load_segmentation_dataset():
    path_train = "UCI_datasets/segmentation/segmentation.data"
    path_test = "UCI_datasets/segmentation/segmentation.test"

    data_train = np.array(pd.read_csv(path_train, header=None, delimiter=","))
    data_test = np.array(pd.read_csv(path_test, header=None, delimiter=","))

    data = np.vstack((data_train, data_test))
    data_y, data_x = data[:, 0], data[:, 1:]

    classes = np.unique(data_y)
    dic_str_to_int = dict(zip(classes, np.arange(0, len(classes), 1)))
    data_y = [dic_str_to_int[string] for string in data_y]

    return data_x, np.array(data_y)


def load_data(dataset):
    if dataset == "wine":
        data_x, data_y = load_wine_dataset()

    elif dataset == "dermatology":
        data_x, data_y = load_dermatology_dataset()
        data_y = data_y - 1

    elif dataset == "ionosphere":
        data_x, data_y = load_ionosphere_dataset()

    elif dataset == "libras":
        data_x, data_y = load_libras_dataset()

    elif dataset == "segmentation":
        data_x, data_y = load_segmentation_dataset()

    return data_x, data_y
