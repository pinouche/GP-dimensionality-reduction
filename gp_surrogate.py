from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler

from util import k_fold_valifation_accuracy_rf
from util import match_trees


def multi_tree_gp_surrogate_model(data_x, low_dim_x, test_data_x, test_data_y, seed, share_multi_tree, use_interpretability_model=False,
                                  fitness="autoencoder_teacher_fitness", stacked_gp=False, num_of_layers=1):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    test_data_x = scaler.transform(test_data_x)

    if stacked_gp:
        num_of_blocks = 0
        for layer in range(num_of_layers):
            print("COMPUTING FOR LAYER: " + str(layer))
            building_blocks_train, building_blocks_test = get_building_blocks(data_x, low_dim_x, test_data_x, num_of_blocks,
                                                                              use_interpretability_model, fitness)
            print(building_blocks_train.shape, building_blocks_test.shape)

            if len(building_blocks_train.shape) == 3:
                for index in range(building_blocks_train.shape[0]):
                    data_x = np.hstack((data_x, building_blocks_train[index]))
                    test_data_x = np.hstack((test_data_x, building_blocks_test[index]))
                num_of_blocks += building_blocks_train.shape[0]*2
            else:
                data_x = np.hstack((data_x, building_blocks_train))
                test_data_x = np.hstack((test_data_x, building_blocks_test))
                num_of_blocks += 2

            print("the number of blocks is: " + str(num_of_blocks))

    # Prepare NSGP settings
    if share_multi_tree:
        init_max_tree_height = 3
        num_sub_functions = np.sqrt(data_x.shape[1])+1
    else:
        init_max_tree_height = 7
        num_sub_functions = 0

    if fitness == "autoencoder_teacher_fitness":
        use_linear_scaling = True
    else:
        use_linear_scaling = False

    estimator = NSGP(pop_size=1000, max_generations=2, verbose=True, max_tree_size=100,
                     crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=init_max_tree_height, tournament_size=2, use_linear_scaling=use_linear_scaling,
                     use_erc=False, use_interpretability_model=use_interpretability_model,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
                     fitness=fitness,
                     num_sub_functions=num_sub_functions)

    estimator.fit(data_x, low_dim_x)

    front = estimator.nsgp_.latest_front
    front_non_duplicate = []
    front_string_format = []
    for individual in front:
        if individual.GetHumanExpression() not in front_string_format:
            front_string_format.append(individual.GetHumanExpression())
            front_non_duplicate.append(individual)

    low_dim = []
    individuals = []
    len_programs = []
    fitness = []

    print("duplicate front length: " + str(len(front)) + " , non-duplicate front length: " + str(len(front_non_duplicate)))

    for individual in front_non_duplicate:
        output = individual.GetOutput(test_data_x.astype(float))
        individual_output = np.empty(output.shape)
        for i in range(individual.num_sup_functions):

            scaled_output = individual.sup_functions[i].ls_a + individual.sup_functions[i].ls_b * output[:, i]
            individual_output[:, i] = scaled_output

        low_dim.append(individual_output)
        individuals.append(individual)
        len_programs.append(individual.objectives[1])
        fitness.append(individual.objectives[0])

    low_dim = np.array(low_dim)
    len_programs = np.array(len_programs)
    individuals = np.array(individuals)
    fitness = np.array(fitness)

    # get the indices sorted by the first objective
    indice_array = np.arange(0, len(fitness), 1)
    zipped_list = list(zip(fitness, indice_array))
    zipped_list.sort()
    indices = [val[1] for val in zipped_list]
    # reorder
    low_dim = low_dim[indices]
    len_programs = len_programs[indices]
    individuals = individuals[indices]

    accuracy_list = []
    for index in range(low_dim.shape[0]):
        x = low_dim[index]

        avg_acc, std_acc = k_fold_valifation_accuracy_rf(x, test_data_y, seed)
        accuracy_list.append(avg_acc)

    return accuracy_list, len_programs, individuals


def gp_surrogate_model(data_x, low_dim_x, test_data_x, test_data_y, seed, use_interpretability_model=False):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    test_data_x = scaler.transform(test_data_x)

    print("I AM OUT OF THE LOOP")

    num_latent_dimensions = low_dim_x.shape[1]
    low_dim = [[] for _ in range(num_latent_dimensions)]
    len_programs = [[] for _ in range(num_latent_dimensions)]
    fitness_programs = [[] for _ in range(num_latent_dimensions)]
    individuals = [[] for _ in range(num_latent_dimensions)]

    for index in range(num_latent_dimensions):

        estimator = NSGP(pop_size=1000, max_generations=2, verbose=True, max_tree_size=100,
                         crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                         initialization_max_tree_height=7, tournament_size=2, use_linear_scaling=True,
                         use_erc=False, use_interpretability_model=use_interpretability_model,
                         functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                         fitness="autoencoder_teacher_fitness",
                         use_multi_tree=False)

        estimator.fit(data_x, low_dim_x[:, index])
        front = estimator.nsgp_.latest_front
        front_non_duplicate = []
        front_string_format = []
        for individual in front:
            if individual.GetHumanExpression() not in front_string_format:
                front_string_format.append(individual.GetHumanExpression())
                front_non_duplicate.append(individual)

        print("duplicate front length: " + str(len(front)) + " , non-duplicate front length: " + str(len(front_non_duplicate)))
        for individual in front_non_duplicate:
            output = individual.GetOutput(test_data_x)
            output = individual.ls_a + individual.ls_b * output

            low_dim[index].append(output)
            len_programs[index].append(individual.objectives[1])
            fitness_programs[index].append(individual.objectives[0])
            individuals[index].append(individual)

    low_dim = np.array(low_dim)
    len_programs = np.array(len_programs)
    individuals = np.array(individuals)

    gp_data, len_programs, individuals = match_trees(low_dim, len_programs, fitness_programs, individuals, num_latent_dimensions)

    accuracy_list = []
    for index in range(gp_data.shape[1]):
        x = gp_data[:, index, :]
        x = np.transpose(x)

        avg_acc, std_acc = k_fold_valifation_accuracy_rf(x, test_data_y, seed)
        accuracy_list.append(avg_acc)

    return accuracy_list, np.transpose(len_programs), np.transpose(individuals)


def get_building_blocks(data_x, low_dim_x, test_data_x, num_blocks, use_interpretability_model=False, fitness="autoencoder_teacher_fitness"):

    num_sub_functions = 0

    if fitness == "autoencoder_teacher_fitness":
        use_linear_scaling = True
    else:
        use_linear_scaling = False

    estimator = NSGP(pop_size=1000, max_generations=2, verbose=True, max_tree_size=5*(num_blocks+1),
                     crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=3, tournament_size=2, use_linear_scaling=use_linear_scaling,
                     use_erc=False, use_interpretability_model=use_interpretability_model,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
                     fitness=fitness,
                     num_sub_functions=num_sub_functions)

    estimator.fit(data_x, low_dim_x)

    front = estimator.nsgp_.latest_front
    front_non_duplicate = []
    front_string_format = []
    for individual in front:
        if individual.GetHumanExpression() not in front_string_format:
            front_string_format.append(individual.GetHumanExpression())
            front_non_duplicate.append(individual)

    print("duplicate front length: " + str(len(front)) + " , non-duplicate front length: " + str(len(front_non_duplicate)))

    building_blocks_train = gp_multi_tree_output(front_non_duplicate, data_x)
    building_blocks_test = gp_multi_tree_output(front_non_duplicate, test_data_x)

    return building_blocks_train, building_blocks_test


def gp_multi_tree_output(front, x):

    low_dim = []
    for individual in front:
        output = individual.GetOutput(x.astype(float))
        individual_output = np.empty(output.shape)
        for i in range(individual.num_sup_functions):
            scaled_output = individual.sup_functions[i].ls_a + individual.sup_functions[i].ls_b * output[:, i]
            individual_output[:, i] = scaled_output

        low_dim.append(individual_output)

    low_dim = np.squeeze(np.array(low_dim))

    return low_dim
