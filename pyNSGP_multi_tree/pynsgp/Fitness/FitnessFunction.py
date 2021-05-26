import numpy as np
from copy import deepcopy
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
import keras
from sklearn.preprocessing import StandardScaler

from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode
from pynsgp.Nodes.MultiTreeRepresentation import MultiTreeIndividual


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train, X_test, y_test, use_linear_scaling=True, second_objective="length",
                 fitness="autoencoder_teacher_fitness"):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.use_linear_scaling = use_linear_scaling
        self.second_objective = second_objective
        self.fitness = fitness
        self.elite = None
        self.evaluations = 0

    def Evaluate(self, individual):

        self.evaluations = self.evaluations + 1
        individual.objectives = []

        if "manifold_fitness" in self.fitness:
            obj1_train = self.stress_cost(self.X_train, individual, 128)
            obj1_test = self.stress_cost(self.X_test, individual, 128)
        elif self.fitness == "neural_decoder_fitness":
            obj1_train = self.neural_decoder_fitness(self.X_train, individual, self.evaluations)
            obj1_test = self.neural_decoder_fitness(self.X_test, individual, self.evaluations)
        elif self.fitness == "autoencoder_teacher_fitness" or self.fitness == "gp_autoencoder_fitness":
            obj1_train = self.EvaluateMeanSquaredError(self.X_train, self.y_train, individual, True)
            obj1_test = self.EvaluateMeanSquaredError(self.X_test, self.y_test, individual, False)

        individual.objectives.append((obj1_train, obj1_test))

        if self.second_objective == "length":
            obj2 = self.EvaluateLength(individual)
        elif self.second_objective == "phi_model":
            obj2 = self.EvaluatePHIsModel(individual)

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
    def stress_cost(self, data, individual, batch_size=128):

        assert batch_size <= self.X_train.shape[0]

        random.seed(self.evaluations)
        indices_vector = random.sample(range(data.shape[0]), batch_size)

        # compute distances on the original data
        similarity_matrix_batch = pdist(data[indices_vector], 'euclidean')

        prediction_batch = data[indices_vector]
        output = individual.GetOutput(prediction_batch)
        # compute distances on the gp predictions (lower dimensional data)
        similarity_matrix_pred = pdist(output, 'euclidean')

        if self.fitness == "manifold_fitness_absolute":
            fitness = np.sum(np.abs(similarity_matrix_batch - similarity_matrix_pred))
        elif self.fitness == "manifold_fitness_rank_footrule":

            full_similirarity_matrix_org = squareform(similarity_matrix_batch)
            full_similirarity_matrix_pred = squareform(similarity_matrix_pred)

            ranks_true = np.argsort(full_similirarity_matrix_org, axis=1)
            ranks_pred = np.argsort(full_similirarity_matrix_pred, axis=1)

            fitness = np.sum(np.abs(ranks_true-ranks_pred))

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

    # fitness function that trains a decoder to use as the fitness
    def neural_decoder_fitness(self, data, individual, seed):

        output = individual.GetOutput(data)

        scaler = StandardScaler()
        output = scaler.fit_transform(output)

        input_size = data.shape[1]
        latent_size = output.shape[1]
        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(int((input_size + latent_size) / 4), activation="elu", use_bias=True,
                               trainable=True, kernel_initializer=initializer, input_shape=(latent_size,)),
            # latent_layer
            keras.layers.Dense(int((input_size + latent_size) / 2), activation="elu", use_bias=True,
                               trainable=True, kernel_initializer=initializer),

            keras.layers.Dense(input_size, activation=keras.activations.linear, use_bias=False,
                               trainable=True, kernel_initializer=initializer)
        ])

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])

        model_info = model.fit(output, data, batch_size=32, epochs=200, verbose=False)

        argmin = np.argmin(model_info.history["loss"])
        loss = np.mean(model_info.history["loss"][argmin-1:argmin+1])

        keras.backend.clear_session()

        if loss == np.nan:
            print("TRUE")
            loss = np.inf

        return loss

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

    def __EvaluatePHIsModelOfNormalTree(self, individual):
        subtree = individual.GetSubtree()
        n_nodes = len(subtree)
        n_ops = 0
        n_naops = 0
        n_vars = 0
        dimensions = set()
        n_constants = 0
        for n in subtree:
            if n.arity > 0:
                n_ops += 1
                if n.is_not_arithmetic:
                    n_naops += 1
            else:
                str_repr = str(n)
                if str_repr[0] == 'x':
                    n_vars += 1
                    idx = int(str_repr[1:len(str_repr)])
                    dimensions.add(idx)
                else:
                    n_constants += 1
        n_nacomp = individual.Count_n_nacomp()
        n_dim = len(dimensions)

        '''
                print('-------------------')
                print(subtree)
                print('nodes:',n_nodes)
                print('dimensions', n_dim)
                print('variables', n_vars)
                print('constants', n_constants)
                print('ops', n_ops)
                print('naops', n_naops)
                print('nacomp', n_nacomp)
                print('------------------')
                '''

        result = self._ComputeInterpretabilityScore(n_dim, n_vars,
                                                    n_constants, n_nodes, n_ops, n_naops, n_nacomp)
        result = -1 * result

        return result

    def __EvaluatePHIsModelOfMultiTree(self, individual):
        '''
                we have two options here, one is to assume that the user can understand
                the parts, and then the total from them. In that case, we just compute
                phi for each sub_function and each sup_function.

                The other would be that, instead, each sup_function must be interpreted as a whole
                of itself + sub_functions. 
                To implement that, we can create a temp sup_function where, each time we find a FeatureNode, 
                we replace that with a clone of the sub_function it represents.
                
                I assume people are smart and go with the first option.
                '''
        phis = list()
        for sup_fun in individual.sup_functions:
            partial_phi = self.__EvaluatePHIsModelOfNormalTree(sup_fun)
            phis.append(partial_phi)
        for sub_fun in individual.sub_functions:
            partial_phi = self.__EvaluatePHIsModelOfNormalTree(sub_fun)
            phis.append(partial_phi)
        phi = np.sum(phis)
        return phi

    def EvaluatePHIsModel(self, individual):
        if isinstance(individual, MultiTreeIndividual):
            phi = self.__EvaluatePHIsModelOfMultiTree(individual)
        else:
            phi = self.__EvaluatePHIsModelOfNormalTree(individual)

        return phi

    def _ComputeInterpretabilityScore(self, n_dim, n_vars, n_const, n_nodes, n_ops, na_ops, na_comp):
        # correctness weighted by confidence:
        features = [n_nodes, n_ops, na_ops, na_comp]
        coeffs = [-0.00195041, -0.00502375, -0.03351907, -0.04472121]
        result = np.sum(np.multiply(features, coeffs)) * 100
        return result
