from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

from sklearn.preprocessing import StandardScaler

from util import plot_low_dim
from util import k_fold_valifation_accuracy_rf


def gp_surrogate_model(data_x, low_dim_x, data_y, num_latent_dimensions, seed, dataset, method, share_multi_tree):

    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    print("I AM OUT OF THE LOOP")

    low_dim = []
    len_programs = []
    individuals = []

    # Prepare NSGP settings
    if share_multi_tree:
        init_max_tree_height = 3
        num_sub_functions = np.sqrt(data_x.shape[1])+1
    else:
        init_max_tree_height = 7
        num_sub_functions = 0

    estimator = NSGP(pop_size=1000, max_generations=1, verbose=True, max_tree_size=300,
                     crossover_rate=0.9, mutation_rate=0.1, op_mutation_rate=0.1, min_depth=2,
                     initialization_max_tree_height=init_max_tree_height, tournament_size=2, use_linear_scaling=True,
                     use_erc=False, use_interpretability_model=False,
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

    print("duplicate front length: " + str(len(front)) + " , non-duplicate front length: " + str(len(front_non_duplicate)))
    for individual in front_non_duplicate:
        output = individual.GetOutput(data_x)
        individual_output = np.empty(output.shape)
        for i in range(individual.num_sup_functions):

            scaled_output = individual.sup_functions[i].ls_a + individual.sup_functions[i].ls_b * output[:, i]
            individual_output[:, i] = scaled_output

        low_dim.append(individual_output)
        len_programs.append(individual.objectives[1])
        individuals.append(individual)

    low_dim = np.array(low_dim)
    len_programs = np.array(len_programs)

    #if num_latent_dimensions == 2:
    #    path = "gecco/" + dataset + "/" + method + "/"
    #    name_save_fig = path + dataset + "_" + "gp_" + str(seed)
    #    plot_low_dim(low_dim[:, 0, :], data_y, name_save_fig)

    accuracy_list = []
    for index in range(low_dim.shape[0]):
        x = low_dim[index]

        avg_acc, std_acc = k_fold_valifation_accuracy_rf(x, data_y, seed)
        accuracy_list.append(avg_acc)

    return accuracy_list, len_programs, individuals
