# Libraries
import numpy as np 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SymbolicRegressionNodes import *
from pynsgp.Fitness.FitnessFunction import SymbolicRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP

from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True )
y = np.vstack([y,y]).transpose()


use_multi_tree = False
if len(y.shape) > 1 and y.shape[1] > 1:
  use_multi_tree = True

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )


# Prepare NSGP settings
nsgp = NSGP(pop_size=100, max_generations=100, verbose=True, max_tree_size=32, 
	crossover_rate=0.334, mutation_rate=0.333, op_mutation_rate=0.333, min_depth=2, initialization_max_tree_height=6, 
	tournament_size=2, use_linear_scaling=True, use_erc=True, use_interpretability_model=True,
  use_multi_tree=use_multi_tree, num_sub_functions=int(np.sqrt(X_train.shape[1])+1), # idk, hyper-parameter
	functions = [AddNode(), SubNode(), MulNode(), DivNode()])

# Fit like any sklearn estimator
nsgp.fit(X_train,y_train)

# Obtain the front of non-dominated solutions (according to the training set)
front = nsgp.get_front()
print('len front:',len(front))
for solution in front:
  print(solution.GetHumanExpression())
  break