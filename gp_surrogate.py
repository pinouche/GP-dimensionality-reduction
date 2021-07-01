from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler


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
                     pop_size=pop_size, max_generations=5, verbose=True, max_tree_size=50,
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


