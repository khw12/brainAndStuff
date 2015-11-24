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
  print "Did not import"
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

# ------------------------------------------------------------------------

#==============================================================================
# ## Plot membrane potentials
# fig = plt.figure(1)
# plt.subplot(211)
# plt.plot(range(T), v1)
# plt.title('Population 1 membrane potentials')
# plt.ylabel('Voltage (mV)')
# plt.ylim([-90, 40])
# 
# plt.subplot(212)
# plt.plot(range(T), v2)
# plt.title('Population 2 membrane potentials')
# plt.ylabel('Voltage (mV)')
# plt.ylim([-90, 40])
# plt.xlabel('Time (ms)')
# 
# path = os.path.join(DIR_PATH, 'membrane_potential.svg') # file name and path
# fig.savefig(path) 
# 
# ## Plot recovery variable
# plt.figure(2)
# plt.subplot(211)
# plt.plot(range(T), u1)
# plt.title('Population 1 recovery variables')
# plt.ylabel('Voltage (mV)')
# 
# plt.subplot(212)
# plt.plot(range(T), u2)
# plt.title('Population 2 recovery variables')
# plt.ylabel('Voltage (mV)')
# plt.xlabel('Time (ms)')
# 
# ## Raster plots of firings
# figure3 = plt.figure(3)
# if firings1.size != 0:
#   plt.subplot(211)
#   plt.scatter(firings1[:, 0], firings1[:, 1] + 1, marker='.')
#   plt.xlim(0, T)
#   plt.ylabel('Neuron number')
#   plt.ylim(0, NUM_EXCITORY+1)
#   plt.title('Population 1 firings')
# 
# if firings2.size != 0:
#   plt.subplot(212)
#   plt.scatter(firings2[:, 0], firings2[:, 1] + 1, marker='.')
#   plt.xlim(0, T)
#   plt.ylabel('Neuron number')
#   plt.ylim(0, NUM_INHIBITORY+1)
#   plt.xlabel('Time (ms)')
#   plt.title('Population 2 firings')
# 
# path = os.path.join(DIR_PATH, 'firings.svg') # file name and path
# fig.savefig(path)
# 
# ## Mean firing rate
# figure4 = plt.figure(4)
# if firings1.size != 0:
#   plt.plot(mean_time, mean_firings)
#   plt.ylabel('Mean firing rate')
#   plt.title('Mean firing rate')
# 
# path = os.path.join(DIR_PATH, 'mean_firing.svg') # file name and path
# figure4.savefig(path)
# 
# plt.show()
# 
# 
#==============================================================================
