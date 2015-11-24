from Izhikevich import ConnectIzhikevichNetworkLayers,GenerateNetwork, IzhikevichModularNetwork, RewireModularNetwork
from jpype import *
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
REPEATS = 8
T = 1000 *2

if __name__ == '__main__':
  rewire_probs = rn.uniform(0,1,REPEATS)
  mean_firings_res = []
  try:
    pool = Pool(8)
    function_arg = itertools.izip(itertools.repeat(T),rewire_probs,itertools.repeat(2))
    mean_firings_res = pool.map(simulation_wrapper_star, function_arg)
  except:
    for p in np.nditer(rewire_probs):
      res = simulation_wrapper(T,p,2)
      mean_firings_res.append(res)

# ------------------------------------------------------------------------
# JIDT integration calculation
#Start JVM
startJVM(getDefaultJVMPath(), "-Djava.class.path=" + "infodynamics.jar")
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
teCalc = teCalcClass()
#atexit.register(shutdownJVM)    #doesn't catch system error

integration_result = []

for [mean_firings, p] in mean_firings_res:
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
 plt.ylim(0, 5)
 plt.ylabel('Integration(bits)')
 plt.xlim(0, 1)
 plt.xlabel('Rewiring probability p')
 plt.title('Integration')
#path = os.path.join(DIR_PATH, 'integration.svg')
#fig3.savefig(path)
 
plt.show()
# ------------------------------------------------------------------------