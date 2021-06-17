import numpy as np
from copy import deepcopy
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau

from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode
from pynsgp.Nodes.MultiTreeRepresentation import MultiTreeIndividual


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train, X_test, y_test, train_data_x_pca, test_data_x_pca, use_linear_scaling=True, second_objective="length",
                 fitness="autoencoder_teacher_fitness"):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_data_x_pca = train_data_x_pca
        self.test_data_x_pca = test_data_x_pca
        self.use_linear_scaling = use_linear_scaling
        self.second_objective = second_objective
        self.fitness = fitness
        self.elite = None
        self.evaluations = 0

    def Evaluate(self, individual):

        self.evaluations = self.evaluations + 1
        individual.objectives = []

        if "manifold_fitness" in self.fitness:
            obj1_train = self.stress_cost(self.X_train, self.train_data_x_pca, individual, 64)
            obj1_test = self.stress_cost(self.X_test, self.test_data_x_pca, individual, 64)
        elif self.fitness == "autoencoder_teacher_fitness" or self.fitness == "gp_autoencoder_fitness":
            obj1_train = self.EvaluateMeanSquaredError(self.X_train, self.y_train, individual, True)
            obj1_test = self.EvaluateMeanSquaredError(self.X_test, self.y_test, individual, False)

        individual.objectives.append((obj1_train, obj1_test))

        if self.second_objective == "length":
            obj2 = self.EvaluateLength(individual)

        individual.objectives.append(obj2)

        if not self.elite or individual.objectives[0][0] < self.elite.objectives[0][0]:
            del self.elite
            self.elite = deepcopy(individual)

    def __EvaluateMeanSquaredErrorOfNormalTree(self, data_x, data_y, individual, train):
        output = individual.GetOutput(data_x)

        a = 0.0
        b = 1.0
        if self.use_linear_scaling:
            if train:
                b = np.cov(self.y_train, output)[0, 1] / (np.var(output) + 1e-10)
                a = np.mean(self.y_train) - b * np.mean(output)
                individual.ls_a = a
                individual.ls_b = b
            else:
                a = individual.ls_a
                b = individual.ls_b
        scaled_output = a + b * output
        fit_error = np.mean(np.square(data_y - scaled_output))

        return fit_error

    def __EvaluateMeanSquaredErrorOfMultiTree(self, data_x, data_y, individual, train):
        # compute multi-output, starting from sub_functions
        output = individual.GetOutput(data_x)
        fit_errors = list()
        for i in range(individual.num_sup_functions):

            a = 0.0
            b = 1.0
            if self.use_linear_scaling:
                if train:
                    b = np.cov(self.y_train[:, i], output[:, i])[0, 1] / (np.var(output[:, i]) + 1e-10)
                    a = np.mean(self.y_train[:, i]) - b * np.mean(output[:, i])
                    individual.sup_functions[i].ls_a = a
                    individual.sup_functions[i].ls_b = b
                else:
                    a = individual.sup_functions[i].ls_a
                    b = individual.sup_functions[i].ls_b

            scaled_output = a + b * output[:, i]
            fit_error = np.mean(np.square(data_y[:, i] - scaled_output))
            fit_errors.append(fit_error)

        fit_error = np.mean(fit_errors)
        return fit_error

    # fitness function to directly evolve trees to do dimensionality reduction
    def stress_cost(self, data, data_pca, individual, batch_size=64):

        assert batch_size <= self.X_train.shape[0]

        random.seed(self.evaluations)
        indices_vector = random.sample(range(data.shape[0]), batch_size)

        # compute distances on the original data (pca)
        similarity_matrix_batch = pdist(data_pca[indices_vector], 'euclidean')

        prediction_batch = data[indices_vector]
        output = individual.GetOutput(prediction_batch)
        # compute distances on the gp predictions (lower dimensional data)
        similarity_matrix_pred = pdist(output, 'euclidean')

        if self.fitness == "manifold_fitness_absolute":
            fitness = np.sum(np.abs(similarity_matrix_batch - similarity_matrix_pred))

        elif self.fitness == "manifold_fitness_rank_spearman":

            full_similirarity_matrix_org = squareform(similarity_matrix_batch)
            full_similirarity_matrix_pred = squareform(similarity_matrix_pred)

            fitness = 0
            for index in range(batch_size):
                corr = kendalltau(full_similirarity_matrix_org[index], full_similirarity_matrix_pred[index])[0]*-1
                if np.isnan(corr):
                    corr = 1
                fitness += corr

            fitness /= batch_size

        return fitness

    def EvaluateMeanSquaredError(self, data_x, data_y, individual, train=True):
        if isinstance(individual, MultiTreeIndividual):
            fit_error = self.__EvaluateMeanSquaredErrorOfMultiTree(data_x, data_y, individual, train)
        else:
            fit_error = self.__EvaluateMeanSquaredErrorOfNormalTree(data_x, data_y, individual, train)

        return fit_error

    def EvaluateLength(self, individual):
        l = 0
        if isinstance(individual, MultiTreeIndividual):
            # precompute lengths of subfunctions
            len_subfunctions = [len(x.GetSubtree()) for x in individual.sub_functions]
            if self.fitness == "gp_autoencoder_fitness":
                l = np.sum(len_subfunctions)
            else:
                for sup_function in individual.sup_functions:
                    for node in sup_function.GetSubtree():
                        if isinstance(node, FeatureNode) and individual.num_sub_functions > 0:
                            # fetch length of sub-function
                            l += len_subfunctions[node.id]
                        else:
                            # count one
                            l += 1
        else:
            l = len(individual.GetSubtree())
        return l

