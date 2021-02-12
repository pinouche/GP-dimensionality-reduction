from sklearn import datasets
import numpy as np
import pandas as pd


# dermatology dataset
def load_ionosphere_dataset():
    path = "../../Documents/UCI datasets/ionosphere/ionosphere.data"

    data = np.array(pd.read_csv(path, header=None, delimiter=","))

    data_y, data_x = data[:, 0].astype(int), data[:, 1:]
    data_x = np.delete(data_x, -1, 1)

    return data_x, data_y


# wine dataset
def load_wine_dataset():
    data_dic = datasets.load_wine()
    data_x_original, data_y_original = data_dic["data"], data_dic["target"]

    return data_x_original, data_y_original


# segmentation dataset
def load_segmentation_dataset():
    path_train = "../../Documents/UCI datasets/segmentation/segmentation.data"
    path_test = "../../Documents/UCI datasets/segmentation/segmentation.test"

    data_train = np.array(pd.read_csv(path_train, header=None, delimiter=","))
    data_test = np.array(pd.read_csv(path_test, header=None, delimiter=","))

    data = np.vstack((data_train, data_test))
    data_y, data_x = data[:, 0], data[:, 1:]

    classes = np.unique(data_y)
    dic_str_to_int = dict(zip(classes, np.arange(0, len(classes), 1)))
    data_y = [dic_str_to_int[string] for string in data_y]

    return data_x, np.array(data_y)


# madelon dataset
def load_madelon():
    def load_madelon_dataset(path_x, path_y):
        data_x = np.array(pd.read_csv(path_x, header=None, delimiter=","))
        data_x_list = []

        for index in range(data_x.shape[0]):
            data_x_list.append([int(val) for val in data_x[index][0].split(" ")[:-1]])

        data_y = np.array(pd.read_csv(path_y, header=None, delimiter=","))

        data_y[data_y == -1] = 0

        return np.array(data_x_list), np.squeeze(data_y)

    data_x_train, data_y_train = load_madelon_dataset("../../Documents/UCI datasets/madelon/madelon_train.data",
                                                      "../../Documents/UCI datasets/madelon/madelon_train.labels")

    data_x_val, data_y_val = load_madelon_dataset("../../Documents/UCI datasets/madelon/madelon_valid.data",
                                                  "../../Documents/UCI datasets/madelon/madelon_valid.labels")

    print(data_y_train.shape, data_y_val.shape)

    data_x = np.vstack((data_x_train, data_x_val))
    data_y = np.concatenate([data_y_train, data_y_val])

    return data_x, data_y


def shuffle_data(x_data, y_data, seed):
    np.random.seed(seed)
    shuffle_list = np.arange(x_data.shape[0])
    np.random.shuffle(shuffle_list)
    x_data = x_data[shuffle_list]
    y_data = y_data[shuffle_list]

    return x_data, y_data


def load_data(dataset):
    if dataset == "wine":
        data_x, data_y = load_wine_dataset()

    elif dataset == "ionosphere":
        data_x, data_y = load_ionosphere_dataset()

    elif dataset == "segmentation":
        data_x, data_y = load_segmentation_dataset()

    elif dataset == "madelon":
        data_x, data_y = load_madelon()

    return data_x, data_y
