from Izhikevich import ConnectIzhikevichNetworkLayers,GenerateNetwork, IzhikevichModularNetwork, RewireModularNetwork
from Run import RunSimulation,simulation_wrapper_star,simulation_wrapper

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
    
# try to import modules for multiprocessing
try:
  from multiprocessing import Pool
  import itertools
except:
  pass    
    
# ------------------------------------------------------------------------
# simulation 
T = 1000 

if __name__ == '__main__':
  rewire_probs = np.arange(0,0.6,0.1)
  mean_firings_res = []
  try:
    pool = Pool(8)
    function_arg = itertools.izip(itertools.repeat(T),rewire_probs,itertools.repeat(1),itertools.repeat(0), itertools.repeat(True))
    mean_firings_res = pool.map(simulation_wrapper_star, function_arg)
  except:
    for p in np.nditer(rewire_probs):
      res = simulation_wrapper(T,p,1,0,True)
      mean_firings_res.append(res)