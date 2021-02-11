from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import inspect

from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP


class pyNSGPEstimator(BaseEstimator, RegressorMixin):

	def __init__(self, 
		pop_size=100, 
		max_generations=100, 
		max_evaluations=-1,
		max_time=-1,
		functions=[ AddNode(), SubNode(), MulNode(), DivNode() ], 
		use_erc=True,
		crossover_rate=0.9,
		mutation_rate=0.1,
		op_mutation_rate=1.0,
		initialization_max_tree_height=6,
		min_depth=2,
		tournament_size=4,
		max_tree_size=100,
		use_linear_scaling=True,
		use_interpretability_model=False,
		verbose=False,
		use_multi_tree = False,
		num_sub_functions = 4
		):

		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		for arg, val in values.items():
			setattr(self, arg, val)


	def fit(self, X, y):

		# Check that X and y have correct shape
		#X, y = check_X_y(X, y)
		self.X_ = X
		self.y_ = y
		
		fitness_function = SymbolicRegressionFitness( X, y, self.use_linear_scaling, 
			use_interpretability_model=self.use_interpretability_model )
		
		terminals = []
		if self.use_erc:
			terminals.append( EphemeralRandomConstantNode() )
		n_features = X.shape[1]
		for i in range(n_features):
			terminals.append(FeatureNode(i))


		self.num_sup_functions = 0
		if self.use_multi_tree:
			self.num_sup_functions = y.shape[1]


		nsgp = pyNSGP(fitness_function, 
			self.functions, terminals, 
			pop_size=self.pop_size, 
			max_generations=self.max_generations,
			max_time = self.max_time,
			max_evaluations = self.max_evaluations,
			crossover_rate=self.crossover_rate,
			mutation_rate=self.mutation_rate,
			op_mutation_rate=self.op_mutation_rate,
			initialization_max_tree_height=self.initialization_max_tree_height,
			min_depth=self.min_depth,
			max_tree_size=self.max_tree_size,
			tournament_size=self.tournament_size,
			use_multi_tree = self.use_multi_tree,
			num_sub_functions=self.num_sub_functions,
			num_sup_functions=self.num_sup_functions,
			verbose=self.verbose)

		nsgp.Run()
		self.nsgp_ = nsgp

		return self

	def predict(self, X):
		assert(not self.use_multi_tree) # not implemented!
		# Check fit has been called
		check_is_fitted(self, ['nsgp_'])

		# Input validation
		X = check_array(X)
		fifu = self.nsgp_.fitness_function
		prediction = fifu.elite.ls_a + fifu.elite.elite.ls_b * fifu.elite.GetOutput( X )

		return prediction

	def score(self, X, y=None):
		assert(not self.use_multi_tree) # not implemented!
		if y is None:
			raise ValueError('The ground truth y was not set')
		
		# Check fit has been called
		prediction = self.predict(X)
		return -1.0 * np.mean(np.square(y - prediction))

	def get_params(self, deep=True):
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		attributes = [a for a in attributes if not (a[0].endswith('_') or a[0].startswith('_'))]

		dic = {}
		for a in attributes:
			dic[a[0]] = a[1]

		return dic

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_elitist_obj1(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.fitness_function.elite

	def get_front(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.latest_front

	def get_population(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.population