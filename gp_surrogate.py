from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from util import match_trees
from util import plot_low_dim
from util import k_fold_valifation_accuracy_rf


def get_building_blocks(data_x_copy, low_dim_x, num_latent_dimensions, building_blocks_sep, meta_building_blocks_individuals):

    building_blocks = [[] for _ in range(num_latent_dimensions)]

    if building_blocks_sep is None:
        meta_building_blocks_individuals = [[] for _ in range(num_latent_dimensions)]

    number_building_blocks = 0
    for index in range(num_latent_dimensions):
        building_blocks_individuals = [None] * data_x_copy.shape[1]

        if building_blocks_sep is not None:
            data_x = np.hstack((data_x_copy, np.transpose(building_blocks_sep[index])))
            building_blocks_individuals = meta_building_blocks_individuals[index]
        else:
            data_x = data_x_copy

        # Prepare NSGP settings
        estimator = NSGP(pop_size=1000, max_generations=100, verbose=True, max_tree_size=5,
                         crossover_rate=0.9, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                         initialization_max_tree_height=3, tournament_size=2, use_linear_scaling=True,
                         use_erc=False, use_interpretability_model=False,
                         functions = [AddNode(), SubNode(), MulNode(), DivNode()],
                         feature_node_list = building_blocks_individuals)

        estimator.fit(data_x, low_dim_x[:,index])
        front = estimator.nsgp_.latest_front
        front_non_duplicate = []
        front_string_format = []
        for individual in front:
            if individual.GetHumanExpression() not in front_string_format:
                front_string_format.append(individual.GetHumanExpression())
                front_non_duplicate.append(individual)

        print("duplicate front length: " + str(len(front)) + " , non-duplicate front length: " + str(len(front_non_duplicate)))
        number_building_blocks += len(front_non_duplicate)

        for individual in front_non_duplicate:
            output = individual.GetOutput(data_x)
            output = individual.ls_a + individual.ls_b * output

            building_blocks[index].append(output)
            building_blocks_individuals.append(individual)

        meta_building_blocks_individuals[index] = building_blocks_individuals

    return building_blocks, meta_building_blocks_individuals


def gp_surrogate_model(data_x, low_dim_x, data_y, num_latent_dimensions, seed, dataset, method, deep_gp=False, number_layers=2):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    data_x_copy = deepcopy(data_x)

    # here decide whether we want to use common building blocks or not
    if deep_gp:
        building_blocks_sep = None
        building_blocks_individuals = None
        for l in range(number_layers):
            building_blocks_sep, building_blocks_individuals = get_building_blocks(data_x_copy, low_dim_x,
                                                                                   num_latent_dimensions,
                                                                                   building_blocks_sep,
                                                                                   building_blocks_individuals)

    print("I AM OUT OF THE LOOP")

    low_dim = [[] for _ in range(num_latent_dimensions)]
    len_programs = [[] for _ in range(num_latent_dimensions)]
    fitness_programs = [[] for _ in range(num_latent_dimensions)]
    individuals = [[] for _ in range(num_latent_dimensions)]

    for index in range(num_latent_dimensions):

        feature_node_list = [None] * data_x_copy.shape[1]
        if deep_gp:
            data_x = np.hstack((data_x_copy, np.transpose(building_blocks_sep[index])))
            feature_node_list = building_blocks_individuals[index]
            max_tree_height = 4
        else:
            max_tree_height = 7

        # Prepare NSGP settings
        estimator = NSGP(pop_size=1000, max_generations=100, verbose=True, max_tree_size=300,
                         crossover_rate=0.9, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                         initialization_max_tree_height=max_tree_height, tournament_size=2, use_linear_scaling=True,
                         use_erc=False, use_interpretability_model=False,
                         functions=[AddNode(), SubNode(), MulNode(), DivNode()],
                         feature_node_list=feature_node_list)

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
            output = individual.GetOutput(data_x)
            output = individual.ls_a + individual.ls_b * output

            low_dim[index].append(output)
            len_programs[index].append(len(individual.GetSubtree()))
            fitness_programs[index].append(mean_absolute_error(output, low_dim_x[:, index]))
            individuals[index].append(individual)

    low_dim = np.array(low_dim)
    len_programs = np.array(len_programs)

    gp_data, len_programs, individuals = match_trees(low_dim, len_programs, fitness_programs, individuals, num_latent_dimensions)

    if num_latent_dimensions == 2:
        path = "gecco/" + dataset + "/" + method + "/"
        name_save_fig = path + dataset + "_" + "gp_" + str(seed)
        plot_low_dim(gp_data[:, 0, :], data_y, name_save_fig)

    len_programs = np.mean(len_programs, axis=0)

    accuracy_list = []
    for index in range(gp_data.shape[1]):
        x = gp_data[:, index, :]
        x = np.transpose(x)

        avg_acc, std_acc = k_fold_valifation_accuracy_rf(x, data_y, seed, num_fold=10)
        accuracy_list.append(avg_acc)

    return accuracy_list, len_programs, individuals
