from pynsgp.Nodes.SymbolicRegressionNodes import FeatureNode
from pynsgp.Variation import Variation

class MultiTreeIndividual:
  
  def __init__(self, num_sup_functions, num_sub_functions):
    self.num_sup_functions = num_sup_functions
    self.num_sub_functions = num_sub_functions

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
    