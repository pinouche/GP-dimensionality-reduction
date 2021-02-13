from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler

from util import k_fold_valifation_accuracy_rf
from util import match_trees


def multi_tree_gp_surrogate_model(data_x, low_dim_x, test_data_x, test_data_y, seed, share_multi_tree, use_interpretability_model=False):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    test_data_x = scaler.transform(test_data_x)

    print("I AM OUT OF THE LOOP")

    # Prepare NSGP settings
    if share_multi_tree:
        init_max_tree_height = 3
        num_sub_functions = np.sqrt(data_x.shape[1])+1
    else:
        init_max_tree_height = 7
        num_sub_functions = 0

    estimator = NSGP(pop_size=1000, max_generations=100, verbose=True, max_tree_size=100,
                     crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=init_max_tree_height, tournament_size=2, use_linear_scaling=True,
                     use_erc=False, use_interpretability_model=use_interpretability_model,
                     functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                     use_multi_tree=True,
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

        estimator = NSGP(pop_size=1000, max_generations=20, verbose=True, max_tree_size=100,
                         crossover_rate=0.8, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                         initialization_max_tree_height=7, tournament_size=2, use_linear_scaling=True,
                         use_erc=False, use_interpretability_model=use_interpretability_model,
                         functions=[AddNode(), SubNode(), MulNode(), DivNode()],
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
