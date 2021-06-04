import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy
import keras

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor

from pynsgp.Variation import Variation
from pynsgp.Selection import Selection

from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode, EphemeralRandomConstantNode

from pynsgp.Nodes.MultiTreeRepresentation import MultiTreeIndividual


class pyNSGP:

    def __init__(
        self,
        fitness_function,
        functions,
        terminals,
        x_train,
        y_train,
        x_test,
        y_test,
        pop_size=500,
        crossover_rate=0.9,
        mutation_rate=0.1,
        op_mutation_rate=1.0,
        max_evaluations=-1,
        max_generations=-1,
        max_time=-1,
        initialization_max_tree_height=4,
        min_depth=2,
        max_tree_size=100,
        tournament_size=4,
        use_multi_tree=False,
        multi_objective=False,
        num_sub_functions=4,
        num_sup_functions=1,
        verbose=False
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.pop_size = pop_size
        self.fitness_function = fitness_function
        self.functions = functions
        self.terminals = terminals
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.op_mutation_rate = op_mutation_rate

        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.max_time = max_time

        self.initialization_max_tree_height = initialization_max_tree_height
        self.min_depth = min_depth
        assert (min_depth <= initialization_max_tree_height)
        self.max_tree_size = max_tree_size
        self.tournament_size = tournament_size
        self.multi_objective = multi_objective

        self.generations = 0

        self.use_multi_tree = use_multi_tree
        self.num_sub_functions = num_sub_functions
        self.num_sup_functions = num_sup_functions

        if self.use_multi_tree:
            # the terminals of the sup_functions are the sub_functions
            self.supfun_terminals = list()
            for i in range(int(self.num_sub_functions)):
                self.supfun_terminals.append(FeatureNode(i))

            # if num_sub_functions is 0, then the sup_functions use the original terminal set
            if self.num_sub_functions == 0:
                self.supfun_terminals = self.terminals
            else:
                # add ephemeral random constants if they were also in the terminals
                for node in self.terminals:
                    if isinstance(node, EphemeralRandomConstantNode):
                        self.supfun_terminals.append(EphemeralRandomConstantNode())

        self.verbose = verbose

    def __ShouldTerminate(self):
        must_terminate = False
        elapsed_time = time.time() - self.start_time
        if 0 < self.max_evaluations <= self.fitness_function.evaluations:
            must_terminate = True
        elif 0 < self.max_generations <= self.generations:
            must_terminate = True
        elif 0 < self.max_time <= elapsed_time:
            must_terminate = True

        if must_terminate and self.verbose:
            print('Terminating at\n\t',
                  self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time, 2), 'seconds')

        return must_terminate

    def Run(self):

        self.start_time = time.time()

        # range(2): the first list is for training set and the second list is for test set
        if self.use_multi_tree:
            list_info = [[] for _ in range(2)]
        else:
            list_info = []

        self.population = []

        # ramped half-n-half
        curr_max_depth = self.min_depth
        init_depth_interval = self.pop_size / (self.initialization_max_tree_height - self.min_depth + 1)
        next_depth_interval = init_depth_interval

        for i in range(self.pop_size):
            if i >= next_depth_interval:
                next_depth_interval += init_depth_interval
                curr_max_depth += 1

            if self.use_multi_tree:
                g = MultiTreeIndividual(self.num_sup_functions, self.num_sub_functions)
                g.InitializeRandom(
                    self.functions, self.supfun_terminals,
                    self.functions, self.terminals,
                    method='grow',
                    max_supfun_height=curr_max_depth, min_supfun_height=self.min_depth,
                    max_subfun_height=curr_max_depth, min_subfun_height=self.min_depth,
                )

                f = MultiTreeIndividual(self.num_sup_functions, self.num_sub_functions)
                f.InitializeRandom(
                    self.functions, self.supfun_terminals,
                    self.functions, self.terminals,
                    method='full',
                    max_supfun_height=curr_max_depth, min_supfun_height=self.min_depth,
                    max_subfun_height=curr_max_depth, min_subfun_height=self.min_depth,
                )

            else:
                g = Variation.GenerateRandomTree(self.functions, self.terminals, curr_max_depth, curr_height=0, method='grow',
                                                 min_depth=self.min_depth)
                f = Variation.GenerateRandomTree(self.functions, self.terminals, curr_max_depth, curr_height=0, method='full',
                                                 min_depth=self.min_depth)

            self.fitness_function.Evaluate(g)
            self.population.append(g)
            self.fitness_function.Evaluate(f)
            self.population.append(f)

        while not self.__ShouldTerminate():

            selected = Selection.tournament(self.population, self.pop_size, self.multi_objective, tournament_size=self.tournament_size)

            O = []
            for i in range(self.pop_size):

                o = deepcopy(selected[i])

                variation_event_happened = False

                # variation of multi trees
                if isinstance(o, MultiTreeIndividual):
                    
                    while not variation_event_happened:

                        for i in range(o.num_sub_functions):
                            if random() < self.crossover_rate:
                                o.sub_functions[i] = Variation.SubtreeCrossover(o.sub_functions[i], selected[randint(self.pop_size)].sub_functions[i])
                                variation_event_happened = True
                            elif random() < self.mutation_rate:
                                o.sub_functions[i] = Variation.SubtreeMutation(o.sub_functions[i], self.functions, self.terminals,
                                                                               max_height=self.initialization_max_tree_height)
                                variation_event_happened = True
                            elif random() < self.op_mutation_rate:
                                o.sub_functions[i] = Variation.OnePointMutation(o.sub_functions[i], self.functions, self.terminals)
                                variation_event_happened = True

                            # correct for violation of constraints
                            if len(o.sub_functions[i].GetSubtree()) > self.max_tree_size:
                                o.sub_functions[i] = deepcopy(selected[i].sub_functions[i])

                        # for the sup functions, we want each head/function to represent the same output dimension
                        for i in range(o.num_sup_functions):
                            if random() < self.crossover_rate:
                                o.sup_functions[i] = Variation.SubtreeCrossover(o.sup_functions[i], selected[randint(self.pop_size)].sup_functions[i])
                                variation_event_happened = True
                            elif random() < self.mutation_rate:
                                o.sup_functions[i] = Variation.SubtreeMutation(o.sup_functions[i], self.functions, self.supfun_terminals,
                                                                               max_height=self.initialization_max_tree_height)
                                variation_event_happened = True
                            elif random() < self.op_mutation_rate:
                                o.sup_functions[i] = Variation.OnePointMutation(o.sup_functions[i], self.functions, self.supfun_terminals)
                                variation_event_happened = True

                            if self.fitness_function.EvaluateLength(o) > self.max_tree_size:
                                o.sup_functions[i] = deepcopy(selected[i].sup_functions[i])

                    self.fitness_function.Evaluate(o)

                # variation of normal individuals
                else:
                    while not variation_event_happened:
                        if random() < self.crossover_rate:
                            o = Variation.SubtreeCrossover(o, selected[randint(self.pop_size)])
                            variation_event_happened = True
                        elif random() < self.mutation_rate:
                            o = Variation.SubtreeMutation(o, self.functions, self.terminals, max_height=self.initialization_max_tree_height)
                            variation_event_happened = True
                        elif random() < self.op_mutation_rate:
                            o = Variation.OnePointMutation(o, self.functions, self.terminals)
                            variation_event_happened = True

                    if len(o.GetSubtree()) > self.max_tree_size:
                        o = deepcopy(selected[i])
                    else:
                        self.fitness_function.Evaluate(o)

                O.append(o)

            if self.multi_objective:
                PO = self.population + O

                new_population = []
                fronts = self.FastNonDominatedSorting(PO)
                self.latest_front = deepcopy(fronts[0])

                curr_front_idx = 0
                while curr_front_idx < len(fronts) and len(fronts[curr_front_idx]) + len(new_population) <= self.pop_size:
                    self.ComputeCrowdingDistances(fronts[curr_front_idx])
                    for p in fronts[curr_front_idx]:
                        new_population.append(p)
                    curr_front_idx += 1

                if len(new_population) < self.pop_size:
                    # fill in remaining
                    self.ComputeCrowdingDistances(fronts[curr_front_idx])
                    fronts[curr_front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)

                    while len(fronts[curr_front_idx]) > 0 and len(new_population) < self.pop_size:
                        new_population.append(fronts[curr_front_idx][0])  # pop first because they were sorted in desc order
                        fronts[curr_front_idx].pop(0)

                    # clean up leftovers
                    while len(fronts[curr_front_idx]) > 0:
                        del fronts[curr_front_idx][0]

            else:
                new_population = deepcopy(O)
                self.latest_front = new_population

            if self.verbose:
                print('g:', self.generations, 'elite obj1:', np.round(self.fitness_function.elite.objectives[0][0], 3),
                      ', obj2:', np.round(self.fitness_function.elite.objectives[1], 3))
                print('elite:', self.fitness_function.elite.GetHumanExpression())

            # compute information from the champion HERE
            if self.use_multi_tree:
                elite = self.fitness_function.elite

                accuracy_champ_train, len_champ_train, tree_champ, x_low_train = self.get_information_from_front([elite], self.x_train, self.y_train)
                accuracy_champ_test, len_champ_test, tree_champ, x_low_test = self.get_information_from_front([elite], self.x_test, self.y_test)

                reconstruction_train_loss, reconstruction_test_loss = self.reconstruction_multi_output(x_low_train, x_low_test)

                if tree_champ.num_sub_functions > 0:
                    tree_champ = tree_champ.sub_functions

                print("TRAIN: ", self.fitness_function.elite.objectives[0][0], "TEST: ", self.fitness_function.elite.objectives[0][1])
                list_info[0].append((self.fitness_function.elite.objectives[0][0], accuracy_champ_train, reconstruction_train_loss, len_champ_train,
                                     tree_champ))
                list_info[1].append((self.fitness_function.elite.objectives[0][1], accuracy_champ_test, reconstruction_test_loss, len_champ_train,
                                     tree_champ))

                if self.generations == self.max_generations - 1:
                    if self.multi_objective:
                        front_non_duplicate = self.get_non_duplicate_front(self.latest_front)

                        front_information = []
                        for individual in front_non_duplicate:
                            accuracy_train, length, tree, x_low_train = self.get_information_from_front([individual], self.x_train, self.y_train)
                            accuracy_test, length, tree, x_low_test = self.get_information_from_front([individual], self.x_test, self.y_test)
                            reconstruction_train_loss, reconstruction_test_loss = self.reconstruction_multi_output(x_low_train, x_low_test)

                            if tree.num_sub_functions > 0:
                                tree = tree.sub_functions

                            front_information.append((accuracy_test, reconstruction_test_loss, length, tree))

                        self.front_information = front_information
                    else:
                        self.front_information = None

            else:
                list_info.append(self.fitness_function.elite)

            self.population = new_population
            self.generations = self.generations + 1

        self.info_generations = list_info

    def FastNonDominatedSorting(self, population):
        rank_counter = 0
        nondominated_fronts = []
        dominated_individuals = {}
        domination_counts = {}
        current_front = []

        for i in range(len(population)):
            p = population[i]

            dominated_individuals[p] = []
            domination_counts[p] = 0

            for j in range(len(population)):
                if i == j:
                    continue
                q = population[j]

                if p.Dominates(q):
                    dominated_individuals[p].append(q)
                elif q.Dominates(p):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                p.rank = rank_counter
                current_front.append(p)

        while len(current_front) > 0:
            next_front = []
            for p in current_front:
                for q in dominated_individuals[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        q.rank = rank_counter + 1
                        next_front.append(q)
            nondominated_fronts.append(current_front)
            rank_counter += 1
            current_front = next_front

        already_seen = set()
        discard_front = []
        for p in population:
            representation = p.GetHumanExpression()
            if representation not in already_seen:
                already_seen.add(representation)
            else:
                # p must be a duplicate then
                # find where p is, remove it from the front it was assigned to, give it a bad rank
                for i, q in enumerate(nondominated_fronts[p.rank]):
                    if nondominated_fronts[p.rank][i] == p:
                        nondominated_fronts[p.rank].pop(i)
                        break
                p.rank = np.inf
                discard_front.append(p)
        # put front with duplicates at the end of the list (it will be considered only if the cumulative size of the previous fronts < pop.size)
        if len(discard_front) > 0:
            nondominated_fronts.append(discard_front)
            # filter out fronts that became empty
            nondominated_fronts = [front for front in nondominated_fronts if len(front) > 0]

        return nondominated_fronts

    def ComputeCrowdingDistances(self, front):
        number_of_objs = len(front[0].objectives)
        front_size = len(front)

        for p in front:
            p.crowding_distance = 0

        for i in range(number_of_objs):
            front.sort(key=lambda x: x.objectives[i], reverse=False)

            front[0].crowding_distance = front[-1].crowding_distance = np.inf

            min_obj = front[0].objectives[i]
            max_obj = front[-1].objectives[i]

            if min_obj == max_obj:
                continue

            for j in range(1, front_size - 1):

                if np.isinf(front[j].crowding_distance):
                    # if extrema from previous sorting
                    continue

                prev_obj = front[j - 1].objectives[i]
                next_obj = front[j + 1].objectives[i]

                front[j].crowding_distance += (next_obj - prev_obj) / (max_obj - min_obj)

    def gp_multi_tree_output(self, front, x):

        low_dim = []
        individuals = []
        len_programs = []
        fitness_list = []

        for individual in front:

            if self.fitness_function.fitness != "gp_autoencoder_fitness":
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

    def get_non_duplicate_front(self, front):

        front_non_duplicate = []
        front_string_format = []
        for individual in front:
            if individual.GetHumanExpression() not in front_string_format:
                front_string_format.append(individual.GetHumanExpression())
                front_non_duplicate.append(individual)

        return front_non_duplicate

    def get_information_from_front(self, front, x, y):

        low_dim, individuals, len_programs, fitness_list = self.gp_multi_tree_output(front, x)

        # only want information from the champion
        x_low = low_dim[0]
        length = len_programs[0]
        champion_representation = individuals[0]

        avg_acc, _ = self.k_fold_valifation_accuracy_rf(x_low, y)

        return avg_acc, length, champion_representation, x_low

    def k_fold_valifation_accuracy_rf(self, data_x, data_y, n_splits=5):
        accuracy_list = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_indices, val_indices in kf.split(data_x):
            x_train, y_train = data_x[train_indices], data_y[train_indices]
            x_val, y_val = data_x[val_indices], data_y[val_indices]

            scaler = StandardScaler()
            scaler = scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)

            classifier = RandomForestClassifier()
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_val)

            accuracy = balanced_accuracy_score(y_val, predictions)
            accuracy_list.append(accuracy)

        return np.mean(accuracy_list), np.std(accuracy_list)

    # fitness function that trains a decoder to use as the fitness
    # def neural_decoder_fitness(self, x_low_train, x_low_test):
    #
    #     scaler = StandardScaler()
    #     scaler.fit(x_low_train)
    #     x_low_train = scaler.transform(x_low_train)
    #     x_low_test = scaler.transform(x_low_test)
    #
    #     x_train_org = self.x_train
    #     x_test_org = self.x_test
    #
    #     scaler = StandardScaler()
    #     scaler.fit(x_train_org)
    #     x_train_org = scaler.transform(x_train_org)
    #     x_test_org = scaler.transform(x_test_org)
    #
    #     input_size = x_train_org.shape[1]
    #     latent_size = x_low_train.shape[1]
    #     initializer = keras.initializers.glorot_normal()
    #
    #     model = keras.models.Sequential([
    #
    #         keras.layers.Dense(int((input_size + latent_size) / 4), activation="elu", use_bias=True,
    #                            trainable=True, kernel_initializer=initializer),
    #
    #         keras.layers.Dense(int((input_size + latent_size) / 2), activation="elu", use_bias=True,
    #                            trainable=True, kernel_initializer=initializer),
    #
    #         keras.layers.Dense(input_size, activation=keras.activations.linear, use_bias=False,
    #                            trainable=True, kernel_initializer=initializer)
    #     ])
    #
    #     adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #     model.compile(optimizer=adam, loss='mse', metrics=['mse'])
    #
    #     model_info = model.fit(x_low_train, x_train_org, batch_size=32, epochs=200, verbose=False, validation_data=(x_low_test, x_test_org))
    #     training_loss = model_info.history["loss"][-1]
    #     test_loss = model_info.history["val_loss"][-1]
    #
    #     keras.backend.clear_session()
    #
    #     return training_loss, test_loss

    def reconstruction_multi_output(self,  x_low_train, x_low_test):

        scaler = StandardScaler()
        scaler.fit(x_low_train)
        x_low_train = scaler.transform(x_low_train)
        x_low_test = scaler.transform(x_low_test)

        x_train = self.x_train
        x_test = self.x_test

        model = KernelRidge(kernel='poly', degree=2)
        est = MultiOutputRegressor(model)
        est.fit(x_low_train, x_train)
        preds_train = est.predict(x_low_train)
        preds_test = est.predict(x_low_test)

        train_reconstruction_error = np.mean((preds_train - x_train) ** 2)
        test_reconstruction_error = np.mean((preds_test - x_test) ** 2)

        return train_reconstruction_error, test_reconstruction_error
