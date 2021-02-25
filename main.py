import multiprocessing
import pickle, os

from autoencoder import get_hidden_layers
from util import train_base_model
from util import k_fold_valifation_accuracy_rf
from gp_surrogate import multi_tree_gp_surrogate_model
from gp_surrogate import gp_surrogate_model
from load_data import load_data
from load_data import shuffle_data


def low_dim_accuracy(dataset, seed, data_struc, num_latent_dimensions=2, share_multi_tree=False, use_phi=False, manifold_fitness=False):
    print("COMPUTING FOR RUN NUMBER: " + str(seed))

    dic_one_run = {}

    split_proportion = [0.6, 0.3, 0.1]

    data_x, data_y = load_data(dataset)
    data_x, data_y = shuffle_data(data_x, data_y, seed)

    # data used for the unsupervised/self-supervised DR algorithms
    base_model_data_x = data_x[:int(data_x.shape[0]*split_proportion[0])]
    # data used to train the gp_surrogate model
    gp_surrogate_data_x = data_x[int(data_x.shape[0]*split_proportion[0]):int(data_x.shape[0]*(split_proportion[1]+split_proportion[0]))]
    # data used to train the random forest (for original data, base model, and gp surrogate model)
    test_data_x, test_data_y = data_x[int(data_x.shape[0]*(1-split_proportion[2])):], data_y[int(data_x.shape[0]*(1-split_proportion[2])):]

    # get the low dimensional representation of the data
    model = train_base_model(base_model_data_x, seed, num_latent_dimensions)

    low_dim_x = get_hidden_layers(model, gp_surrogate_data_x)[3]
    low_dim_test_x = get_hidden_layers(model, test_data_x)[3]

    print("Computing for original dataset")
    org_avg_acc, org_std_acc = k_fold_valifation_accuracy_rf(test_data_x, test_data_y, seed)

    print("Computing for teacher")
    avg_acc, std_acc = k_fold_valifation_accuracy_rf(low_dim_test_x, test_data_y, seed)

    print("Computing for method GP")
    if share_multi_tree is not None:
        accuracy_gp, length_list, individuals = multi_tree_gp_surrogate_model(gp_surrogate_data_x, low_dim_x, test_data_x, test_data_y,
                                                                              seed, share_multi_tree, use_phi, manifold_fitness)
    else:
        accuracy_gp, length_list, individuals = gp_surrogate_model(gp_surrogate_data_x, low_dim_x, test_data_x, test_data_y, seed, use_phi)

    dic_one_run["original_data_accuracy"] = org_avg_acc
    dic_one_run["teacher_accuracy"] = avg_acc
    dic_one_run["gp_accuracy"] = accuracy_gp
    dic_one_run["gp_length"] = length_list

    dic_one_run["champion_individual"] = individuals[0]
    dic_one_run["champion_accuracy"] = accuracy_gp[0]
    dic_one_run["champion_length"] = length_list[0]

    dic_one_run["median_individual"] = individuals[int(len(length_list)*0.5)]
    dic_one_run["median_accuracy"] = accuracy_gp[int(len(length_list) * 0.5)]
    dic_one_run["median_length"] = length_list[int(len(length_list) * 0.5)]

    dic_one_run["75%_individual"] = individuals[int(len(length_list)*0.25)]
    dic_one_run["75%_accuracy"] = accuracy_gp[int(len(length_list)*0.25)]
    dic_one_run["75%_length"] = length_list[int(len(length_list)*0.25)]

    data_struc["run_number_" + str(seed)] = dic_one_run


if __name__ == "__main__":

    num_of_runs = 1

    manifold_fitness = True

    for dataset in ["observatory"]:
        for use_phi in [False]:
            for share_multi_tree in [False]:  # True: shared, multi-tree; False: non-shared, multi-tree; None: vanilla GP (non-shared, not multi-tree)

                if share_multi_tree is None and manifold_fitness:
                    raise ValueError("the GP representation is not multi-tree and the fitness function is manifold function!")

                for num_latent_dimensions in [2]:

                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()

                    p = [multiprocessing.Process(target=low_dim_accuracy, args=(dataset, seed, return_dict, num_latent_dimensions, share_multi_tree,
                                                                                use_phi, manifold_fitness))
                                                                                for seed in range(num_of_runs)]
                    for proc in p:
                        proc.start()
                    for proc in p:
                        proc.join()

                    results = return_dict.values()

                    path = "gecco/" + dataset + "/"

                    os.makedirs(path, exist_ok=True)
                    file_name = path + "results_" + dataset + "_" + str(num_latent_dimensions)

                    if share_multi_tree:
                        file_name = file_name + "_shared"
                    elif not share_multi_tree:
                        file_name = file_name + "_not_shared"
                    elif share_multi_tree is None:
                        file_name = file_name + "_vanilla"

                    if manifold_fitness:
                        file_name = file_name + "_manifold_fitness"

                    if use_phi:
                        file_name = file_name + "_phi"
                    else:
                        file_name = file_name + "_len"

                    pickle.dump(results, open(file_name + ".p", "wb"))
