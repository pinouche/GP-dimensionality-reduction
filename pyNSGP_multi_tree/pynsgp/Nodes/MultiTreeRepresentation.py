from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode
from pynsgp.Variation import Variation

import numpy as np

class MultiTreeIndividual:
  
  def __init__(self, num_sup_functions, num_sub_functions):
    self.num_sup_functions = int(num_sup_functions)
    self.num_sub_functions = int(num_sub_functions)

    self.sup_functions = list()
    self.sub_functions = list()

    self.subfun_terminals = list()
    for i in range(self.num_sub_functions):
      self.subfun_terminals.append(FeatureNode(i))

    # some stuff like in node
    self.objectives = []
    self.rank = 0
    self.crowding_distance = 0


  def Dominates(self, other):
    better_somewhere = False 
    for i in range(len(self.objectives)):
      if self.objectives[i] > other.objectives[i]:
        return False
      if self.objectives[i] < other.objectives[i]:
        better_somewhere = True 

    return better_somewhere

  def GetOutput( self, X ):
    if self.num_sub_functions > 0:
      sub_function_outputs = list()
      for i in range(self.num_sub_functions):
        sub_function_output = self.sub_functions[i].GetOutput(X)
        sub_function_outputs.append(sub_function_output)
      # assemble the output of sub_functions into something usable in FeatureNode
      X_subfun = np.vstack(sub_function_outputs).transpose()
    else:
      X_subfun = X
    # now compute output of sup_functions by re-using the ones of the sub_functions
    sup_fun_outputs = list()
    for i in range(self.num_sup_functions):
      sup_fun_output = self.sup_functions[i].GetOutput(X_subfun)
      sup_fun_outputs.append(sup_fun_output)

    # the final output, should be (n * k) dimensional
    final_output = np.vstack(sup_fun_outputs).transpose()
    return final_output



  def InitializeRandom(self, 
    supfun_functions, supfun_terminals,
    subfun_functions, subfun_terminals,
    method='grow',
    max_supfun_height=6, min_supfun_height=2,
    max_subfun_height=6, min_subfun_height=2
    ):

    for _ in range(self.num_sup_functions):
      fun = Variation.GenerateRandomTree(supfun_functions, supfun_terminals, max_supfun_height, curr_height=0, method=method, min_depth=min_supfun_height)
      self.sup_functions.append(fun)
    
    for _ in range(self.num_sub_functions):
      fun = Variation.GenerateRandomTree(subfun_functions, subfun_terminals, max_subfun_height, curr_height=0, method=method, min_depth=min_subfun_height)
      self.sub_functions.append(fun)

  def GetHumanExpression(self):
    result = ""
    for i in range(self.num_sub_functions):
      result += '\tsubfun'+str(i)+ ': '+ self.sub_functions[i].GetHumanExpression() + ";"
    result += "\t"
    for i in range(self.num_sup_functions):
      human_expr_supfun = self.sup_functions[i].GetHumanExpression()
      if self.num_sub_functions > 0:
        human_expr_supfun = human_expr_supfun.replace('x','subfun')
      result += '\tsupfun'+str(i)+ ': '+ human_expr_supfun + ";"
    return result
    