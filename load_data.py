import numpy as np
import pandas as pd


# segmentation dataset
def load_segmentation_dataset():
    path_train = "./segmentation/segmentation.data"
    path_test = "./segmentation/segmentation.test"

    data_train = np.array(pd.read_csv(path_train, header=None, delimiter=","))
    data_test = np.array(pd.read_csv(path_test, header=None, delimiter=","))

    data = np.vstack((data_train, data_test))
    data_y, data_x = data[:, 0], data[:, 1:]

    classes = np.unique(data_y)
    dic_str_to_int = dict(zip(classes, np.arange(0, len(classes), 1)))
    data_y = [dic_str_to_int[string] for string in data_y]

    return data_x, np.array(data_y)


def load_winequality_dataset():
    path_red_wine = "./wine_quality/winequality-red.csv"
    path_white_wine = "./wine_quality/winequality-white.csv"

    data_red = pd.read_csv(path_red_wine, delimiter=";")
    data_white = pd.read_csv(path_white_wine, delimiter=";")

    data = np.vstack((data_red, data_white))
    data_x = data[:, :-1]
    data_y = [0] * data_red.shape[0] + [1] * data_white.shape[0]

    return data_x, np.array(data_y)


def load_credit_dataset():
    path = "./credit/credit.data"

    data = np.array(pd.read_csv(path, header=None, delimiter=","))

    data_list = []

    for index in range(data.shape[0]):
        row = [float(value) for value in data[index][0].split(" ") if value.isnumeric()]
        data_list.append(row)

    data_list = np.array(data_list)

    data_x = data_list[:, :-1]
    data_y = data_list[:, -1] - 1

    return data_x, data_y


def load_observatory_dataset():
    path = "./observatory/observatory.data"

    data = np.array(pd.read_csv(path, delimiter=","))
    data_y, data_x = data[:, -1], data[:, :-1]

    classes = np.unique(data_y)
    dic_str_to_int = dict(zip(classes, np.arange(0, len(classes), 1)))
    data_y = [dic_str_to_int[string] for string in data_y]

    return data_x, np.array(data_y)


def shuffle_data(x_data, y_data, seed):
    np.random.seed(seed)
    shuffle_list = np.arange(x_data.shape[0])
    np.random.shuffle(shuffle_list)
    x_data = x_data[shuffle_list]
    y_data = y_data[shuffle_list]

    return x_data, y_data


def load_data(dataset):
    if dataset == "winequality":
        data_x, data_y = load_winequality_dataset()

    elif dataset == "credit":
        data_x, data_y = load_credit_dataset()

    elif dataset == "segmentation":
        data_x, data_y = load_segmentation_dataset()

    elif dataset == "observatory":
        data_x, data_y = load_observatory_dataset()

    return data_x, data_y
