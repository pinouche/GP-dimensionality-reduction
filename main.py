import multiprocessing
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from autoencoder import get_hidden_layers
from util import train_base_model
from util import k_fold_valifation_accuracy_rf
from gp_surrogate import multi_tree_gp_surrogate_model
from load_data import load_data
from load_data import shuffle_data

from sklearn.decomposition import PCA


def low_dim_accuracy(dataset, seed, data_struc, num_latent_dimensions, operators_rate, share_multi_tree=False, second_objective="length",
                     fitness="autoencoder_teacher_fitness", pop_size=100, multi_objective=False):
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

    scaler = StandardScaler()
    scaler = scaler.fit(train_data_x)
    train_data_x = scaler.transform(train_data_x)
    test_data_x = scaler.transform(test_data_x)

    # PCA tansformation of the original data
    est = PCA(n_components=train_data_x.shape[1])
    est.fit(train_data_x)
    explained_variance_mask = np.cumsum(est.explained_variance_ratio_) >= 0.95
    num_components = list(explained_variance_mask).index(True)
    train_data_x_pca = est.transform(train_data_x)[:, :num_components]
    test_data_x_pca = est.transform(test_data_x)[:, :num_components]

    print(train_data_x_pca.shape)

    # get the low dimensional representation of the data
    model = train_base_model(train_data_x, train_data_x_pca, seed, num_latent_dimensions)

    low_dim_train_x = get_hidden_layers(model, train_data_x)[3]
    low_dim_test_x = get_hidden_layers(model, test_data_x)[3]

    print("Computing for original dataset")
    accuracy_train_org, accuracy_test_org = k_fold_valifation_accuracy_rf(train_data_x, test_data_x, train_data_y, test_data_y)

    print("Computing for teacher")
    accuracy_train_teacher, accuracy_test_teacher = k_fold_valifation_accuracy_rf(low_dim_train_x, low_dim_test_x, train_data_y, test_data_y)
    avg_reconstruction = model.evaluate(test_data_x.astype('float32'), test_data_x_pca.astype('float32'), verbose=False)[0]

    print("Computing for method GP")
    if share_multi_tree is not None:
        front_last_generation = multi_tree_gp_surrogate_model(train_data_x, low_dim_train_x, train_data_y, test_data_x, low_dim_test_x, test_data_y,
                                                              train_data_x_pca, test_data_x_pca,
                                                              operators_rate,
                                                              share_multi_tree, second_objective, fitness,
                                                              pop_size, multi_objective)
    else:
        pass

    dic_one_run["original_data_accuracy"] = accuracy_test_org
    dic_one_run["teacher_data"] = (accuracy_test_teacher, avg_reconstruction)
    dic_one_run["front_last_generation"] = front_last_generation

    data_struc["run_number_" + str(seed)] = dic_one_run


if __name__ == "__main__":

    multi_objective = False

    crossover_rate = 0.9
    op_mutation_rate = 0.1
    mutation_rate = op_mutation_rate
    operators_rate = (crossover_rate, op_mutation_rate, mutation_rate)

    num_of_runs = 1
    pop_size = 200

    fitness_list = ["manifold_fitness_sammon_euclidean", "manifold_fitness_rank_euclidean", "manifold_fitness_sammon_isomap",
                    "manifold_fitness_rank_isomap", "autoencoder_teacher_fitness", "gp_autoencoder_fitness"]

    # fitness_list = ["autoencoder_teacher_fitness"]

    for dataset in ["credit"]:
        for second_objective in ["length"]:
            for fitness in fitness_list:

                if fitness == "gp_autoencoder_fitness":
                    list_gp_method = [True]  # for gp-autoencoder fitness, we want to use the shared multi-tree GP representation
                else:
                    list_gp_method = [False]

                if list_gp_method:

                    for gp_method in list_gp_method:  # True: shared, multi-tree; False: non-shared, multi-tree; None: vanilla GP
                        print("THE GP METHOD IS")
                        print("fitness: ", fitness, "GP representation: ", gp_method)

                        for num_latent_dimensions in [2]:

                            manager = multiprocessing.Manager()
                            return_dict = manager.dict()

                            p = [multiprocessing.Process(target=low_dim_accuracy,
                                                         args=(dataset, seed, return_dict, num_latent_dimensions, operators_rate, gp_method,
                                                               second_objective, fitness, pop_size, multi_objective))
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

                            file_name = file_name + "_" + fitness + "_" + second_objective

                            file_name = file_name + "_pop=" + str(pop_size)

                            file_name = file_name + "_multi_objective=" + str(multi_objective)

                            # pickle.dump(results, open(file_name + ".p", "wb"))
