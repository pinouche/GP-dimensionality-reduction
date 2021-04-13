import multiprocessing
import pickle
import os

from autoencoder import get_hidden_layers
from util import train_base_model
from util import k_fold_valifation_accuracy_rf
from gp_surrogate import multi_tree_gp_surrogate_model
from gp_surrogate import gp_surrogate_model
from load_data import load_data
from load_data import shuffle_data


def low_dim_accuracy(dataset, seed, data_struc, num_latent_dimensions=2, share_multi_tree=False, use_phi=False, fitness="autoencoder_teacher_fitness",
                     stacked_gp=False, pop_size=100):
    print("COMPUTING FOR RUN NUMBER: " + str(seed))

    dic_one_run = {}

    # train and test
    split_proportion = [0.5, 0.5]

    data_x, data_y = load_data(dataset)
    data_x, data_y = shuffle_data(data_x, data_y, seed)

    # data used for the unsupervised/self-supervised DR algorithms
    train_data_x, train_data_y = data_x[:int(data_x.shape[0] * split_proportion[0])], data_y[:int(data_x.shape[0] * split_proportion[0])]
    # test data
    test_data_x, test_data_y = data_x[int(data_x.shape[0] * split_proportion[0]):], data_y[int(data_x.shape[0] * split_proportion[0]):]

    # get the low dimensional representation of the data
    model = train_base_model(train_data_x, seed, num_latent_dimensions)

    low_dim_x = get_hidden_layers(model, train_data_x)[3]
    low_dim_test_x = get_hidden_layers(model, test_data_x)[3]

    print("Computing for original dataset")
    org_avg_acc, org_std_acc = k_fold_valifation_accuracy_rf(test_data_x, test_data_y)

    print("Computing for teacher")
    avg_acc, std_acc = k_fold_valifation_accuracy_rf(low_dim_test_x, test_data_y)

    print("Computing for method GP")
    if share_multi_tree is not None:
        info, front_last_generation = multi_tree_gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, test_data_y,
                                                                    share_multi_tree, use_phi, fitness,
                                                                    stacked_gp, pop_size)
    else:
        # here, front_last_generation is None
        info, front_last_generation = gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, test_data_y, use_phi, pop_size)

    dic_one_run["original_data_accuracy"] = org_avg_acc
    dic_one_run["teacher_accuracy"] = avg_acc
    dic_one_run["gp_info_generations"] = info
    dic_one_run["front_last_generation"] = front_last_generation

    data_struc["run_number_" + str(seed)] = dic_one_run


if __name__ == "__main__":

    num_of_runs = 1
    pop_size = 100

    fitness_list = ["manifold_fitness_absolute", "manifold_fitness_rank_footrule", "manifold_fitness_rank_spearman",
                    "autoencoder_teacher_fitness", "gp_autoencoder_fitness"]

    for dataset in ["segmentation"]:
        for use_phi in [False]:
            for stacked_gp in [False]:
                for fitness in fitness_list:

                    # specify the allowed combination of methods
                    if stacked_gp and fitness != "gp_autoencoder_fitness":
                        list_gp_method = [False]  # we only want multi-tree non-shared
                    elif stacked_gp and fitness == "gp_autoencoder_fitness":
                        list_gp_method = []  # we do not want to compute for stacked gp and gp_autoencoder fitness (empty list)
                    elif not stacked_gp and fitness == "gp_autoencoder_fitness":
                        list_gp_method = [True]  # for gp-autoencoder fitness, we want to use the shared multi-tree GP representation
                    elif not stacked_gp and fitness == "autoencoder_teacher_fitness":  # we only want vanilla GP when using teacher model
                        list_gp_method = [None, False, True]
                    else:
                        list_gp_method = [False]

                    if list_gp_method:

                        for gp_method in list_gp_method:  # True: shared, multi-tree; False: non-shared, multi-tree; None: vanilla GP
                            print("THE GP METHOD IS")
                            print("stacked_gp: ", stacked_gp, "fitness: ", fitness, "GP representation: ", gp_method)

                            if gp_method is None and ("manifold_fitness" in fitness or fitness == "neural_decoder_fitness"):
                                raise ValueError("the GP representation is not multi-tree and the fitness function is manifold function!")

                            if gp_method is not False and stacked_gp:
                                raise ValueError("we want to to use non-shared multi-tree with stacked GP (stacked GP is already shared)")

                            for num_latent_dimensions in [2]:

                                manager = multiprocessing.Manager()
                                return_dict = manager.dict()

                                p = [multiprocessing.Process(target=low_dim_accuracy,
                                                             args=(dataset, seed, return_dict, num_latent_dimensions, gp_method,
                                                                   use_phi, fitness, stacked_gp, pop_size))
                                     for seed in range(num_of_runs)]

                                for proc in p:
                                    proc.start()
                                for proc in p:
                                    proc.join()

                                results = return_dict.values()

                                path = "gecco/" + dataset + "/"

                                os.makedirs(path, exist_ok=True)
                                file_name = path + "results_" + dataset + "_" + str(num_latent_dimensions)

                                if gp_method:
                                    file_name = file_name + "_mt_shared"
                                elif gp_method is False:
                                    file_name = file_name + "_mt_not_shared"
                                elif gp_method is None:
                                    file_name = file_name + "_vanilla"

                                if stacked_gp:
                                    file_name = file_name + "_stacked"

                                file_name = file_name + "_" + fitness

                                if use_phi:
                                    file_name = file_name + "_phi"
                                else:
                                    file_name = file_name + "_len"

                                file_name = file_name + "_pop=" + str(pop_size)

                                pickle.dump(results, open(file_name + ".p", "wb"))
