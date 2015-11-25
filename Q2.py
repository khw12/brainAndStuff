from Izhikevich import ConnectIzhikevichNetworkLayers,GenerateNetwork, IzhikevichModularNetwork, RewireModularNetwork
from jpype import *
from Run import RunSimulation,simulation_wrapper_star,simulation_wrapper

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import os

# try to import modules for multiprocessing
try:
  from multiprocessing import Pool
  import itertools
except:
  pass


# ------------------------------------------------------------------------
# simulation 
REPEATS = 20    # Number of trials
T = 1000 * 60   # Duration of each trial

if __name__ == '__main__':
  rewire_probs = rn.uniform(0,1,REPEATS)
  mean_firings_res = []
  try:
    pool = Pool(8)
    function_arg = itertools.izip(itertools.repeat(T),rewire_probs,itertools.repeat(2),itertools.repeat(1000), itertools.repeat(False))
    mean_firings_res = pool.map(simulation_wrapper_star, function_arg)
  except:
    for p in np.nditer(rewire_probs):
      res = simulation_wrapper(T,p,2,1000,False)
      mean_firings_res.append(res)

# ------------------------------------------------------------------------
# JIDT integration calculation
#Start JVM
startJVM(getDefaultJVMPath(), "-Djava.class.path=" + "infodynamics.jar")
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
teCalc = teCalcClass()

integration_result = []

for [mean_firings, p] in mean_firings_res:
  print 'Calculating multi-information for p = ' + str(p)
  teCalc.initialise(8)
  teCalc.startAddObservations()
  teCalc.addObservations(mean_firings)
  teCalc.finaliseAddObservations()
  result = teCalc.computeAverageLocalOfObservations()
  integration_result.append([p, result])

shutdownJVM()

## Multi-information/Integration
I = np.array(integration_result)
fig3 = plt.figure()
if len(I) != 0:
 plt.scatter(I[:, 0], I[:, 1], marker='.')
 plt.ylim(0, 6)
 plt.ylabel('Integration(bits)')
 plt.xlim(0, 1)
 plt.xlabel('Rewiring probability p')
 plt.title('Integration')
 
DIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'q2')
if not os.path.exists(DIR_PATH):
  os.makedirs(DIR_PATH)
path = os.path.join(DIR_PATH, 'integration_' + str(REPEATS) + '.svg')
fig3.savefig(path)
 
## Uncomment the following line to view graphs in popup windows
#plt.show()
# ------------------------------------------------------------------------
