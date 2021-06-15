from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor

from util import k_fold_valifation_accuracy_rf


def multi_tree_gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, low_dim_test_x, test_data_y, train_data_x_pca, test_data_x_pca,
                                  operators_rate, share_multi_tree, second_objective="length", fitness="autoencoder_teacher_fitness", pop_size=100,
                                  multi_objective=False):

    scaler = StandardScaler()
    scaler.fit(train_data_x)
    train_data_x = scaler.transform(train_data_x)
    test_data_x = scaler.transform(test_data_x)

    # Prepare NSGP settings
    if share_multi_tree and fitness == "gp_autoencoder_fitness":
        init_max_tree_height = 7
        num_sub_functions = low_dim_x.shape[1]
    else:
        init_max_tree_height = 7
        num_sub_functions = 0

    if fitness == "autoencoder_teacher_fitness" or fitness == "gp_autoencoder_fitness":
        use_linear_scaling = True
    else:
        use_linear_scaling = False

    estimator = NSGP(train_data_x, train_data_y, test_data_x, test_data_y, train_data_x_pca, test_data_x_pca,
                     pop_size=pop_size, max_generations=2, verbose=True, max_tree_size=50,
                     crossover_rate=operators_rate[0], mutation_rate=operators_rate[1], op_mutation_rate=operators_rate[2], min_depth=1,
                     initialization_max_tree_height=init_max_tree_height, tournament_size=2, use_linear_scaling=use_linear_scaling,
                     use_erc=True, second_objective=second_objective,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
                     multi_objective=multi_objective,
                     fitness=fitness,
                     num_sub_functions=num_sub_functions)

    if fitness != "gp_autoencoder_fitness":
        estimator.fit(train_data_x, low_dim_x, test_data_x, low_dim_test_x)
    else:
        estimator.fit(train_data_x, train_data_x_pca, test_data_x, test_data_x_pca)

    front_information = estimator.get_front_info()

    return front_information


# def gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, low_dim_test_x, test_data_y, operators_rate,
#                        second_objective="length", pop_size=100, multi_objective=False):
#
#     scaler = StandardScaler()
#     scaler.fit(train_data_x)
#     train_data_x = scaler.transform(train_data_x)
#     test_data_x = scaler.transform(test_data_x)
#
#     num_latent_dimensions = low_dim_x.shape[1]
#     num_sample_train = train_data_x.shape[0]
#     num_sample_test = test_data_x.shape[0]
#
#     generations = 2
#     low_dim_train_array = np.empty((generations, num_latent_dimensions, num_sample_train))
#     low_dim_test_array = np.empty((generations, num_latent_dimensions, num_sample_test))
#     individuals = [[] for _ in range(num_latent_dimensions)]
#     fitness_train_list = [[] for _ in range(num_latent_dimensions)]
#     fitness_test_list = [[] for _ in range(num_latent_dimensions)]
#
#     for index in range(num_latent_dimensions):
#
#         estimator = NSGP(train_data_x, train_data_y, test_data_x, test_data_y,
#                          pop_size=pop_size, max_generations=generations, verbose=True, max_tree_size=50,
#                          crossover_rate=operators_rate[0], mutation_rate=operators_rate[1], op_mutation_rate=operators_rate[2], min_depth=1,
#                          initialization_max_tree_height=7, tournament_size=2, use_linear_scaling=True,
#                          use_erc=True, second_objective=second_objective,
#                          functions=[AddNode(), SubNode(), MulNode(), DivNode()],
#                          fitness="autoencoder_teacher_fitness",
#                          use_multi_tree=False, multi_objective=multi_objective)
#
#         estimator.fit(train_data_x, low_dim_x[:, index], test_data_x, low_dim_test_x[:, index])
#
#         # here champions refer to front
#         champions = [estimator.get_front_info()[0]]
#         fitness_train_list[index].append([c.objectives[0][0] for c in champions])
#         fitness_test_list[index].append([c.objectives[0][1] for c in champions])
#         individuals[index].append(champions)
#
#         # this is for the champions for each generation
#         low_dim_train = get_single_tree_output(individuals[index][0], train_data_x)
#         low_dim_test = get_single_tree_output(individuals[index][0], test_data_x)
#
#         low_dim_train_array[:, index, :] = low_dim_train
#         low_dim_test_array[:, index, :] = low_dim_test
#
#     fitness_train = np.mean(np.squeeze(np.array(fitness_train_list)), axis=0)
#     fitness_test = np.mean(np.squeeze(np.array(fitness_test_list)), axis=0)
#     individuals = np.squeeze(np.array(individuals))
#     summed_length = np.reshape(np.array([ind.objectives[1] for ind in individuals.flatten()]), individuals.shape)
#     if num_latent_dimensions > 1:
#         summed_length = np.sum(summed_length, axis=0)
#
#     # range(2) is to store information for both train and test
#     info = [[] for _ in range(2)]
#     for index in range(low_dim_train_array.shape[0]):
#         x_train_low, x_test_low = np.transpose(low_dim_train_array[index]), np.transpose(low_dim_test_array[index])
#
#         avg_acc_train, _ = k_fold_valifation_accuracy_rf(x_train_low, train_data_y)
#         avg_acc_test, _ = k_fold_valifation_accuracy_rf(x_test_low, test_data_y)
#         train_reconstrution_loss, test_reconstruction_loss = reconstruction_multi_output(x_train_low, x_test_low, train_data_x, test_data_x)
#
#         info[0].append((fitness_train[index], avg_acc_train, train_reconstrution_loss, summed_length[index], np.transpose(individuals)[index]))
#         info[1].append((fitness_test[index], avg_acc_test, test_reconstruction_loss, summed_length[index], np.transpose(individuals)[index]))
#
#     return info
#
#
# def get_single_tree_output(front, x):
#     low_dim = []
#     for individual in front:
#         output = individual.GetOutput(x)
#         output = individual.ls_a + individual.ls_b * output
#
#         low_dim.append(output)
#
#     low_dim = np.array(low_dim)
#
#     return low_dim


def gp_multi_tree_output(front, x, fitness):

    low_dim = []
    individuals = []
    len_programs = []
    fitness_list = []

    for individual in front:

        if fitness != "gp_autoencoder_fitness":
            output = individual.GetOutput(x.astype(float))
            individual_output = np.empty(output.shape)
            for i in range(individual.num_sup_functions):

                scaled_output = individual.sup_functions[i].ls_a + individual.sup_functions[i].ls_b * output[:, i]
                individual_output[:, i] = scaled_output

        else:
            sub_function_outputs = list()
            for i in range(individual.num_sub_functions):
                sub_function_output = individual.sub_functions[i].GetOutput(x.astype(float))
                sub_function_outputs.append(sub_function_output)
                individual_output = np.vstack(sub_function_outputs).transpose()

        low_dim.append(individual_output)
        individuals.append(individual)
        len_programs.append(individual.objectives[1])
        fitness_list.append(individual.objectives[0])

    return np.array(low_dim), np.array(individuals), np.array(len_programs), np.array(fitness_list)


def get_non_duplicate_front(estimator):

    front = estimator.nsgp_.latest_front
    front_non_duplicate = []
    front_string_format = []
    for individual in front:
        if individual.GetHumanExpression() not in front_string_format:
            front_string_format.append(individual.GetHumanExpression())
            front_non_duplicate.append(individual)

    return front_non_duplicate


def reconstruction_multi_output(x_low_train, x_low_test, x_train, x_test):

    scaler = StandardScaler()
    scaler.fit(x_low_train)
    x_low_train = scaler.transform(x_low_train)
    x_low_test = scaler.transform(x_low_test)

    model = KernelRidge(kernel='poly', degree=2)
    est = MultiOutputRegressor(model)
    est.fit(x_low_train, x_train)
    preds_train = est.predict(x_low_train)
    preds_test = est.predict(x_low_test)

    train_reconstruction_error = np.mean((preds_train - x_train) ** 2)
    test_reconstruction_error = np.mean((preds_test - x_test) ** 2)

    return train_reconstruction_error, test_reconstruction_error

