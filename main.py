import multiprocessing
import pickle
from copy import deepcopy
import numpy as np

from sklearn.model_selection import StratifiedKFold

from util import get_lower_dim
from util import plot_low_dim
from util import k_fold_valifation_accuracy_rf
from gp_surrogate import gp_surrogate_model
from load_data import load_data


def low_dim_accuracy(dataset, method, seed, data_struc, num_latent_dimensions=2, deep_gp=False, number_layers=1):
    print("COMPUTING FOR RUN NUMBER: " + str(seed))

    dic_one_run = {}

    data_x, data_y = load_data(dataset)

    # get the low dimensional representation of the data
    if method == "nn":
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_indices, val_indices in kf.split(deepcopy(data_x), deepcopy(data_y)):
            train_x, val_x = deepcopy(data_x[train_indices]), deepcopy(data_x[val_indices])
            data_y = data_y[np.concatenate((train_indices, val_indices))]
            data_x = data_x[np.concatenate((train_indices, val_indices))]
            break

        low_dim_x, model = get_lower_dim(train_x, seed, num_latent_dimensions, method, val_x)

    else:
        low_dim_x, model = get_lower_dim(data_x, seed, num_latent_dimensions, method)

    # plot the low dimensional representation of the data

    if num_latent_dimensions == 2:
        path = "gecco/" + dataset + "/" + method + "/"
        name_save_fig = path + dataset + "_" + method + "_" + str(seed)
        plot_low_dim(np.transpose(low_dim_x), data_y, name_save_fig)

    print("Computing for original dataset")
    org_avg_acc, org_std_acc = k_fold_valifation_accuracy_rf(data_x, data_y, seed)

    print("Computing for method " + str(method))
    avg_acc, std_acc = k_fold_valifation_accuracy_rf(low_dim_x, data_y, seed)

    print("Computing for method GP")
    accuracy_gp, length_list, individuals = gp_surrogate_model(data_x, low_dim_x, data_y,
                                                               num_latent_dimensions, seed, dataset,
                                                               method, deep_gp, number_layers)

    dic_one_run["original_data_accuracy"] = org_avg_acc
    dic_one_run["teacher_accuracy"] = avg_acc
    dic_one_run["gp_accuracy"] = accuracy_gp
    dic_one_run["gp_length"] = length_list
    dic_one_run["champion_accuracy"] = accuracy_gp[0]
    dic_one_run["champion_length"] = length_list[0]

    data_struc["run_number_" + str(seed)] = dic_one_run


if __name__ == "__main__":

    deep_gp = True
    num_layers = 1

    num_of_runs = 30
    method = "nn"  # nn or pca

    for dataset in ["segmentation"]:
        for deep_gp in [False, True]:
            for num_latent_dimensions in [1, 2, 3, 4, 5]:

                manager = multiprocessing.Manager()
                return_dict = manager.dict()

                p = [multiprocessing.Process(target=low_dim_accuracy, args=(dataset, method, seed,
                                                                            return_dict, num_latent_dimensions,
                                                                            deep_gp, num_layers)) for seed in range(num_of_runs)]

                for proc in p:
                    proc.start()
                for proc in p:
                    proc.join()

                results = return_dict.values()

                path = "gecco/" + dataset + "/" + method + "/"
                file_name = path + "results_" + dataset + "_" + method + "_" + str(num_latent_dimensions)

                if deep_gp:
                    file_name = file_name + "_deep_gp"

                pickle.dump(results, open(file_name + ".p", "wb"))




