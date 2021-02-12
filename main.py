import multiprocessing
import pickle, os
from copy import deepcopy
import numpy as np

from sklearn.model_selection import StratifiedKFold

from autoencoder import get_hidden_layers
from util import train_base_model
from util import k_fold_valifation_accuracy_rf
from gp_surrogate import gp_surrogate_model
from load_data import load_data
from load_data import shuffle_data



def low_dim_accuracy(dataset, method, seed, data_struc, num_latent_dimensions=2, share_multi_tree=False,
    quantile_from_pareto, use_phi):
    print("COMPUTING FOR RUN NUMBER: " + str(seed))

    dic_one_run = {}

    split_proportion = [0.4, 0.4, 0.2]
    assert np.sum(split_proportion) == 1
    data_x, data_y = load_data(dataset)
    data_x, data_y = shuffle_data(data_x, data_y, seed)

    # data used for the unsupervised/self-supervised DR algorithms
    base_model_data_x = data_x[:int(data_x.shape[0]*split_proportion[0])]
    base_model_data_y = data_y[:int(data_x.shape[0]*split_proportion[0])]

    # data used to train the gp_surrogate model
    gp_surrogate_data_x = data_x[int(data_x.shape[0]*split_proportion[0]):int(data_x.shape[0]*split_proportion[1]*2)]

    # data used to train the random forest (for original data, base model, and gp surrogate model)
    test_data_x, test_data_y = data_x[int(data_x.shape[0]*(1-split_proportion[2])):], data_y[int(data_x.shape[0]*(1-split_proportion[2])):]

    # get the low dimensional representation of the data
    if method == "nn":
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_indices, val_indices in kf.split(deepcopy(base_model_data_x), deepcopy(base_model_data_y)):
            train_x, val_x = deepcopy(base_model_data_x[train_indices]), deepcopy(base_model_data_x[val_indices])
            break

        model = train_base_model(train_x, seed, num_latent_dimensions, method, val_x)

        low_dim_x = get_hidden_layers(model, gp_surrogate_data_x)[3]
        low_dim_test_x = get_hidden_layers(model, test_data_x)[3]


    else:
        model = train_base_model(base_model_data_x, seed, num_latent_dimensions, method)
        low_dim_x = model.transform(gp_surrogate_data_x)
        low_dim_test_x = model.transform(test_data_x)

    print("Computing for original dataset")
    org_avg_acc, org_std_acc = k_fold_valifation_accuracy_rf(test_data_x, test_data_y, seed)

    print("Computing for method " + str(method))
    avg_acc, std_acc = k_fold_valifation_accuracy_rf(low_dim_test_x, test_data_y, seed)

    print("Computing for method GP")
    # TODO: now it gets a bit tricky: you use the "surrogate_data" split to train the GP (if you want to "early stop" that, you'd split that thing further into 2)
    # of course the surrogate data goes through the net to get the latent representation used as label for GP.
    # and then you use the "test_data" to train & validate the random forest (of course again report only val_acc)
    # the same validation date of the test set, you use it to compute fidelity (MSE between latent of neural net & surrogate latent)
    accuracy_gp, length_list, individuals = gp_surrogate_model(gp_surrogate_data_x, low_dim_x, test_data_x, test_data_y, seed, share_multi_tree,
        use_interpretability_model=use_phi, quantile_from_pareto=quantile_from_pareto)

    dic_one_run["original_data_accuracy"] = org_avg_acc
    dic_one_run["teacher_accuracy"] = avg_acc
    dic_one_run["gp_accuracy"] = accuracy_gp
    dic_one_run["gp_length"] = length_list
    dic_one_run["champion_accuracy"] = accuracy_gp[0]
    dic_one_run["champion_length"] = length_list[0]

    data_struc["run_number_" + str(seed)] = dic_one_run


if __name__ == "__main__":

    share_multi_tree = False
    num_of_runs = 1
    method = "nn"

    for dataset in ["segmentation"]:

        for quantile_from_pareto in [1.0, 0.5]:

            for use_phi in [True, False]:

                for num_latent_dimensions in [2, 3]:

                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()

                    p = [multiprocessing.Process(target=low_dim_accuracy, args=(dataset, method, seed,
                                                                                return_dict, num_latent_dimensions, share_multi_tree,
                                                                                quantile_from_pareto, use_phi
                                                                                ))
                                                                                for seed in range(num_of_runs)]

                    for proc in p:
                        proc.start()
                    for proc in p:
                        proc.join()

                    results = return_dict.values()

                    path = "gecco/" + dataset + "/" + method + "/"

                    os.makedirs(path, exist_ok=True)
                    file_name = path + "results_" + dataset + "_" + method + "_" + str(num_latent_dimensions)

                    if share_multi_tree:
                        file_name = file_name + "_shared"
                    else:
                        file_name = file_name + "_not_shared"

                    pickle.dump(results, open(file_name + ".p", "wb"))
