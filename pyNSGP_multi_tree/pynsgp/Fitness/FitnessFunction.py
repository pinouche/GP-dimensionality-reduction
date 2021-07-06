import numpy as np
from copy import deepcopy
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, weightedtau
from sklearn import manifold

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
        self.indices_array = 0
        self.data_batch = 0
        self.similarity_matrix_batch = 0

    def Evaluate(self, individual, generations):

        self.evaluations = self.evaluations + 1
        individual.objectives = []

        if "manifold_fitness" in self.fitness:
            obj1_train = self.stress_cost(self.X_train, self.train_data_x_pca, individual, generations, 64)
            obj1_test = self.stress_cost(self.X_test, self.test_data_x_pca, individual, generations, 64)
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
    def stress_cost(self, data, data_pca, individual, generations, batch_size=64):

        assert batch_size <= self.X_train.shape[0]

        if self.evaluations == 1:
            random.seed(generations)
            self.indices_array = random.sample(range(data.shape[0]-1), batch_size)
            self.data_batch = data_pca[self.indices_array]

        prediction_batch = data[self.indices_array]
        output = individual.GetOutput(prediction_batch)

        if "euclidean" in self.fitness:
            # compute distances on the original data (pca)
            if self.evaluations == 1:
                self.similarity_matrix_batch = pdist(self.data_batch, 'euclidean')

            # compute distances on the gp predictions (lower dimensional data)
            similarity_matrix_pred = pdist(output, 'euclidean')

            if "rank" in self.fitness:
                if self.evaluations == 1:
                    self.similarity_matrix_batch = squareform(self.similarity_matrix_batch)
                similarity_matrix_pred = squareform(similarity_matrix_pred)

        else:
            if self.evaluations == 1:
                est = manifold.Isomap(n_neighbors=8)
                est.fit(self.data_batch)
                self.similarity_matrix_batch = est.dist_matrix_

                if "sammon" in self.fitness:
                    self.similarity_matrix_batch = self.similarity_matrix_batch[np.triu_indices(batch_size, 1)]

            est = manifold.Isomap(n_neighbors=8)
            est.fit(output)
            similarity_matrix_pred = est.dist_matrix_

            if "sammon" in self.fitness:
                similarity_matrix_pred = similarity_matrix_pred[np.triu_indices(batch_size, 1)]

        if "sammon" in self.fitness:
            fitness = np.mean(((self.similarity_matrix_batch - similarity_matrix_pred)**2)/(self.similarity_matrix_batch + 1e-4))

        elif "rank" in self.fitness:

            fitness = 0
            for index in range(batch_size):
                corr = weightedtau(self.similarity_matrix_batch[index], similarity_matrix_pred[index])[0]*-1
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

