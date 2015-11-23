from Izhikevic import ConnectIzhikevichNetworkLayers,GenerateNetwork, IzhikevichModularNetwork, RewireModularNetwork
from jpype import *
from Run import RunSimulation

import atexit

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

# fig save path
DIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'q2')
if not os.path.exists(DIR_PATH):
  os.makedirs(DIR_PATH)
    
NUM_NEURONS = 1000
NUM_MODULES = 8
NUM_EXCITORY = 800
NUM_EXCITORY_PER_MODULE = 100
NUM_INHIBITORY = 200
NUM_CONNECTIONS_E_to_E = 1000
NUM_CONNECTIONS_E_to_I = 4

REPEATS = 2
T = 2000

def simulation_wrapper(T,p):
  print 'p is now: ' + str(p)
  Ib = 15    # Base current
  CIJ = IzhikevichModularNetwork(NUM_NEURONS, NUM_MODULES, NUM_EXCITORY_PER_MODULE, NUM_CONNECTIONS_E_to_E, NUM_INHIBITORY)
  net = GenerateNetwork(CIJ, NUM_EXCITORY_PER_MODULE, NUM_INHIBITORY, NUM_EXCITORY, p)

  results = RunSimulation(net, NUM_EXCITORY, NUM_INHIBITORY, T, Ib)
  net = results[0]
  v1 = results[1]
  v2 = results[2]
  u1 = results[3]
  u2 = results[4]

  ## Retrieve firings and add Dirac pulses for presentation
  firings1 = net.layer[0].firings
  firings2 = net.layer[1].firings
  
  if firings1.size != 0:
    v1[firings1[:, 0], firings1[:, 1]] = 30
  
  if firings2.size != 0:
    v2[firings2[:, 0], firings2[:, 1]] = 30
  
  
  ## Mean firing rates
  # note downsampling into intervals of 50ms
  # init var
  INTERVAL = 20
  NUM_SAMPLES = (T-1000)/INTERVAL # discard first second

  mean_firings = np.zeros([NUM_SAMPLES,NUM_MODULES]) # 
  mean_time = range(1000,T,INTERVAL) # start after first second
  
  # note firings is array of array of [t f] where t is timestamp and f is source 
  for [idt,fired] in firings1:
    if idt > 1000: # discard first second
      for window, t_start in enumerate(mean_time):
        if (t_start + 50 > idt) & (t_start <= idt):
          # recover the module from nueron number
          module = fired/NUM_EXCITORY_PER_MODULE
          # no need to filter inhib since its firing1
          mean_firings[window,module] += 1
  
  mean_firings /= 50

  # -------------------------------------------------
  ## Raster plots of firings
  fig1 = plt.figure()
  if firings1.size != 0:
    plt.subplot(211)
    plt.scatter(firings1[:, 0], firings1[:, 1] + 1, marker='.')
    plt.xlim(0, T)
    plt.ylabel('Neuron number')
    plt.ylim(0, NUM_EXCITORY+1)
    plt.title('Population 1 firings')

  if firings2.size != 0:
    plt.subplot(212)
    plt.scatter(firings2[:, 0], firings2[:, 1] + 1, marker='.')
    plt.xlim(0, T)
    plt.ylabel('Neuron number')
    plt.ylim(0, NUM_INHIBITORY+1)
    plt.xlabel('Time (ms)')
    plt.title('Population 2 firings')

  ## Mean firing rate
  fig2 = plt.figure()
  if firings1.size != 0:
    plt.plot(mean_time, mean_firings)
    plt.ylabel('Mean firing rate')
    plt.title('Mean firing rate')

  plt.show()
  # -------------------------------------------------
  return mean_firings, p

def simulation_wrapper_star(T_p):
    return simulation_wrapper(*T_p)


if __name__ == '__main__':
  rewire_probs = rn.uniform(0,1,REPEATS)
  mean_firings_res = []
  try:
    pool = Pool(5)
    function_arg = itertools.izip(itertools.repeat(T),rewire_probs)
    mean_firings_res = pool.map(simulation_wrapper_star, function_arg)
  except:
    for p in np.nditer(rewire_probs):
      res = simulation_wrapper(T,p)
      mean_firings_res.append(res)

# #Start JVM
# startJVM(getDefaultJVMPath(), "-Djava.class.path=" + "infodynamics.jar")
# teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
# teCalc = teCalcClass()
# atexit.register(shutdownJVM)    #doesn't catch system error

# #JIDT
# teCalc.initialise(8)
# teCalc.startAddObservations()
# teCalc.addObservations(mean_firings)
# teCalc.finaliseAddObservations()
# result = teCalc.computeAverageLocalOfObservations()
# res.append([p, result])

# shutdownJVM()

# ## Multi-information/Integration
# I = np.array(res)
# fig3 = plt.figure()
# if len(I) != 0:
#   plt.scatter(I[:, 0], I[:, 1], marker='.')
#   plt.ylim(0, 5)
#   plt.ylabel('Integration(bits)')
#   plt.xlim(0, 1)
#   plt.xlabel('Rewiring probability p')
#   plt.title('Integration')
# path = os.path.join(DIR_PATH, 'integration.svg')
# fig3.savefig(path)