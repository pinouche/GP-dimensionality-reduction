import numpy as np
from copy import deepcopy
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale

from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode
from pynsgp.Nodes.MultiTreeRepresentation import MultiTreeIndividual


class SymbolicRegressionFitness:

    def __init__(self, X_train, y_train, use_linear_scaling=True, use_interpretability_model=False, use_manifold_fitness=False):
        self.X_train = X_train
        self.y_train = y_train
        self.use_linear_scaling = use_linear_scaling
        self.use_interpretability_model = use_interpretability_model
        self.use_manifold_fitness = use_manifold_fitness
        self.elite = None
        self.evaluations = 0

        self.similarity_matrix = None
        if self.use_manifold_fitness:
            self.similarity_matrix = cosine_similarity(scale(self.X_train))

    def Evaluate(self, individual):

        self.evaluations = self.evaluations + 1
        individual.objectives = []

        if self.use_manifold_fitness:
            obj1 = self.stress_cost(individual, 64)
        else:
            obj1 = self.EvaluateMeanSquaredError(individual)

        individual.objectives.append(obj1)

        if self.use_interpretability_model:
            obj2 = self.EvaluatePHIsModel(individual)
        else:
            obj2 = self.EvaluateLength(individual)
        individual.objectives.append(obj2)

        if not self.elite or individual.objectives[0] < self.elite.objectives[0]:
            del self.elite
            self.elite = deepcopy(individual)

    def __EvaluateMeanSquaredErrorOfNormalTree(self, individual):
        output = individual.GetOutput(self.X_train)
        a = 0.0
        b = 1.0
        if self.use_linear_scaling:
            b = np.cov(self.y_train, output)[0, 1] / (np.var(output) + 1e-10)
            a = np.mean(self.y_train) - b * np.mean(output)
            individual.ls_a = a
            individual.ls_b = b
        scaled_output = a + b * output
        fit_error = np.mean(np.square(self.y_train - scaled_output))

        '''
                if str(individual.GetSubtree()) == '[x12]':
                        print(output, a, b, fit_error)
                        print(self.y_train)
                '''

        return fit_error

    def __EvaluateMeanSquaredErrorOfMultiTree(self, individual):
        # quick check that we're not drunk
        assert (self.y_train.shape[1] == individual.num_sup_functions)
        # compute multi-output, starting from sub_functions

        output = individual.GetOutput(self.X_train)
        fit_errors = list()
        for i in range(individual.num_sup_functions):

            a = 0.0
            b = 1.0
            if self.use_linear_scaling:
                b = np.cov(self.y_train[:, i], output[:, i])[0, 1] / (np.var(output[:, i]) + 1e-10)
                a = np.mean(self.y_train[:, i]) - b * np.mean(output[:, i])
                individual.sup_functions[i].ls_a = a
                individual.sup_functions[i].ls_b = b

            scaled_output = a + b * output[:, i]
            fit_error = np.mean(np.square(self.y_train[:, i] - scaled_output))

            '''
                        if str(individual.sup_functions[i].GetSubtree()) == '[x0]' and str(individual.sub_functions[0].GetSubtree()) == '[x12]':
                                print(output, a, b, fit_error)
                                print(self.y_train[:,i])
                                quit()
                        '''
            fit_errors.append(fit_error)
        # now IDK if you want mean or max, I go for mean here
        fit_error = np.mean(fit_errors)
        return fit_error

    # fitness function to directly evolve trees to do dimensionality reduction
    def stress_cost(self, individual, batch_size=64):

        # quick checks that we're not drunk
        assert (self.y_train.shape[1] == individual.num_sup_functions)
        assert batch_size <= self.similarity_matrix.shape[0]

        random.seed(self.evaluations)
        indices_vector = random.sample(range(self.similarity_matrix.shape[0]), batch_size)

        prediction_batch = self.X_train[indices_vector]
        output = individual.GetOutput(prediction_batch)

        similarity_matrix_batch = self.similarity_matrix[indices_vector, indices_vector]
        similarity_matrix_pred = cosine_similarity(output)

        cost = np.sum(np.abs(similarity_matrix_batch - similarity_matrix_pred))

        return cost

    def EvaluateMeanSquaredError(self, individual):
        if isinstance(individual, MultiTreeIndividual):
            fit_error = self.__EvaluateMeanSquaredErrorOfMultiTree(individual)
        else:
            fit_error = self.__EvaluateMeanSquaredErrorOfNormalTree(individual)

        if np.isnan(fit_error):
            fit_error = np.inf

        return fit_error

    def EvaluateLength(self, individual):
        l = 0
        if isinstance(individual, MultiTreeIndividual):
            # precompute lengths of subfunctions
            len_subfunctions = [len(x.GetSubtree()) for x in individual.sub_functions]
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
