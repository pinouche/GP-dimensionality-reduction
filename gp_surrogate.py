from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler

from util import k_fold_valifation_accuracy_rf


def multi_tree_gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, test_data_y, share_multi_tree, use_interpretability_model=False,
                                  fitness="autoencoder_teacher_fitness", stacked_gp=False, num_of_layers=1):

    scaler = StandardScaler()
    scaler.fit(train_data_x)
    train_data_x = scaler.transform(train_data_x)
    test_data_x = scaler.transform(test_data_x)

    if stacked_gp:
        for layer in range(num_of_layers):
            print("COMPUTING FOR LAYER: " + str(layer))
            building_blocks_train, building_blocks_test = get_building_blocks(train_data_x, low_dim_x, test_data_x, use_interpretability_model,
                                                                              fitness)

            for index in range(building_blocks_train.shape[0]):
                train_data_x = np.hstack((train_data_x, building_blocks_train[index]))
                test_data_x = np.hstack((test_data_x, building_blocks_test[index]))
                
            print(train_data_x.shape, building_blocks_train.shape)

    # Prepare NSGP settings
    if share_multi_tree and fitness != "gp_autoencoder_fitness":
        init_max_tree_height = 3
        num_sub_functions = np.sqrt(train_data_x.shape[1])+1
    elif share_multi_tree and fitness == "gp_autoencoder_fitness":
        init_max_tree_height = 3
        num_sub_functions = low_dim_x.shape[1]
    elif stacked_gp:
        init_max_tree_height = 3
        num_sub_functions = 0
    else:
        init_max_tree_height = 7
        num_sub_functions = 0

    if fitness == "autoencoder_teacher_fitness" or fitness == "gp_autoencoder_fitness":
        use_linear_scaling = True
    else:
        use_linear_scaling = False

    estimator = NSGP(train_data_x, train_data_y, test_data_x, test_data_y,
                     pop_size=100, max_generations=2, verbose=True, max_tree_size=100,
                     crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=init_max_tree_height, tournament_size=2, use_linear_scaling=use_linear_scaling,
                     use_erc=False, use_interpretability_model=use_interpretability_model,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
                     fitness=fitness,
                     num_sub_functions=num_sub_functions)

    if fitness != "gp_autoencoder_fitness":
        estimator.fit(train_data_x, low_dim_x)
    else:
        estimator.fit(train_data_x, train_data_x)

    info = estimator.get_list_info()

    return info


def gp_surrogate_model(train_data_x, low_dim_x, train_data_y, test_data_x, test_data_y, seed, use_interpretability_model=False):

    scaler = StandardScaler()
    scaler.fit(train_data_x)
    train_data_x = scaler.transform(train_data_x)
    test_data_x = scaler.transform(test_data_x)

    print(low_dim_x.shape)
    num_latent_dimensions = low_dim_x.shape[1]
    num_sample_train = train_data_x.shape[0]
    num_sample_test = test_data_x.shape[0]

    low_dim_train_array = np.empty((3, num_latent_dimensions, num_sample_train))
    low_dim_test_array = np.empty((3, num_latent_dimensions, num_sample_test))
    individuals = [[] for _ in range(num_latent_dimensions)]

    for index in range(num_latent_dimensions):

        estimator = NSGP(train_data_x, train_data_y, test_data_x, test_data_y,
                         pop_size=100, max_generations=3, verbose=True, max_tree_size=100,
                         crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                         initialization_max_tree_height=7, tournament_size=2, use_linear_scaling=True,
                         use_erc=False, use_interpretability_model=use_interpretability_model,
                         functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                         fitness="autoencoder_teacher_fitness",
                         use_multi_tree=False)

        estimator.fit(train_data_x, low_dim_x[:, index])

        champions = estimator.get_list_info()
        individuals[index].append(champions)

        # this is for the champions for each generation
        low_dim_train = get_single_tree_output(individuals[index][0], train_data_x)
        low_dim_test = get_single_tree_output(individuals[index][0], test_data_x)

        low_dim_train_array[:, index, :] = low_dim_train
        low_dim_test_array[:, index, :] = low_dim_test

    individuals = np.squeeze(np.array(individuals))
    summed_length = np.reshape(np.array([ind.objectives[1] for ind in individuals.flatten()]), individuals.shape)
    if num_latent_dimensions > 1:
        summed_length = np.sum(summed_length, axis=0)

    print(individuals.shape, summed_length.shape, summed_length)

    # get the information here (accuracy, len, individual) for each generation, similarly to multi_tree_output

    # range(2) is to store information for both train and test
    info = [[] for _ in range(2)]
    for index in range(low_dim_train_array.shape[0]):
        x_train, x_test = low_dim_train_array[index], low_dim_test_array[index]
        x_train, x_test = np.transpose(x_train), np.transpose(x_test)

        avg_acc_train, _ = k_fold_valifation_accuracy_rf(x_train, train_data_y, seed)
        avg_acc_test, _ = k_fold_valifation_accuracy_rf(x_test, test_data_y, seed)

        info[0].append((avg_acc_train, summed_length[index], np.transpose(individuals)[index]))
        info[1].append((avg_acc_test, summed_length[index], np.transpose(individuals)[index]))

    return info


def get_building_blocks(data_x, low_dim_x, test_data_x, use_interpretability_model=False, fitness="autoencoder_teacher_fitness"):

    num_sub_functions = 0

    if fitness == "autoencoder_teacher_fitness":
        use_linear_scaling = True
    else:
        use_linear_scaling = False

    estimator = NSGP(pop_size=100, max_generations=2, verbose=True, max_tree_size=100,
                     crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=3, tournament_size=2, use_linear_scaling=use_linear_scaling,
                     use_erc=False, use_interpretability_model=use_interpretability_model,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
                     fitness=fitness,
                     num_sub_functions=num_sub_functions)

    estimator.fit(data_x, low_dim_x)

    front_non_duplicate = get_non_duplicate_front(estimator)
    print("non-duplicate front length: " + str(len(front_non_duplicate)))

    building_blocks_train, _, _, _ = gp_multi_tree_output(front_non_duplicate, data_x)
    building_blocks_test, _, _, _ = gp_multi_tree_output(front_non_duplicate, test_data_x)

    return building_blocks_train, building_blocks_test


def get_single_tree_output(front, x):
    low_dim = []
    for individual in front:
        output = individual.GetOutput(x)
        output = individual.ls_a + individual.ls_b * output

        low_dim.append(output)

    low_dim = np.array(low_dim)

    return low_dim


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
