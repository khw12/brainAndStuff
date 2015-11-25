from Izhikevich import ConnectIzhikevichNetworkLayers,GenerateNetwork, IzhikevichModularNetwork, RewireModularNetwork

import numpy as np
import matplotlib.pyplot as plt
import os 

NUM_NEURONS = 1000
NUM_MODULES = 8
NUM_EXCITORY = 800
NUM_EXCITORY_PER_MODULE = 100
NUM_INHIBITORY = 200
NUM_CONNECTIONS_E_to_E = 1000
NUM_CONNECTIONS_E_to_I = 4
BG_FIRING_PROB = 0.01
Ib = 15    # Base current

def RunSimulation(net, NUM_EXCITORY, NUM_INHIBITORY, T, Ib):
  v1 = np.zeros([T, NUM_EXCITORY])
  v2 = np.zeros([T, NUM_INHIBITORY])
  u1 = np.zeros([T, NUM_EXCITORY])
  u2 = np.zeros([T, NUM_INHIBITORY])

  # SIMULATE
  for t in xrange(T):
    net.layer[0].I = np.zeros(NUM_EXCITORY)
    net.layer[1].I = np.zeros(NUM_INHIBITORY)
    
    # Background firing
    for i in range(NUM_EXCITORY):
      if np.random.poisson(BG_FIRING_PROB) > 0:
        net.layer[0].I[i] = Ib
        
    for i in range(NUM_INHIBITORY):
      if np.random.poisson(BG_FIRING_PROB) > 0:
        net.layer[1].I[i] = Ib
            
    net.Update(t)

    v1[t] = net.layer[0].v
    v2[t] = net.layer[1].v
    u1[t] = net.layer[0].u
    u2[t] = net.layer[1].u

  return([net, v1, v2, u1, u2])
  
  
def simulation_wrapper(T,p,question,discard,save):
  print 'p is now: ' + str(p)

  # fig save path
  if save:
    DIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'q'+str(question),'p'+str(p))
    if not os.path.exists(DIR_PATH):
      os.makedirs(DIR_PATH)
    print 'figure saved at path: ' + DIR_PATH

  CIJ = IzhikevichModularNetwork(NUM_NEURONS, NUM_MODULES, NUM_EXCITORY_PER_MODULE, NUM_CONNECTIONS_E_to_E, NUM_INHIBITORY)
  net = GenerateNetwork(CIJ, NUM_EXCITORY_PER_MODULE, NUM_INHIBITORY, NUM_EXCITORY, p)

  figure = plt.matshow(CIJ[0], cmap=plt.cm.gray, fignum=0)
  if save:
    path = os.path.join(DIR_PATH, 'connectivity_matrix_'+str(p)+'.svg') # file name and path
    plt.savefig(path) 
    
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
  NUM_SAMPLES = (T-discard)/INTERVAL # discard first second
  WINDOW_SIZE = 50
  
  mean_firings = np.zeros([NUM_SAMPLES,NUM_MODULES]) # 
  mean_time = range(discard,T,INTERVAL) # start after first second
  
  # note firings is array of array of [t f] where t is timestamp and f is source 
  for [idt,fired] in firings1:
    if idt > discard: # discard 
      mid_index = (idt-discard)/20
      insert_indices = [mid_index-1,mid_index,mid_index+1]
      for ind in insert_indices:
        if (ind>=0) & (ind<NUM_SAMPLES):
          module = fired/NUM_EXCITORY_PER_MODULE
          mean_firings[ind,module] += 1
  
  mean_firings /= WINDOW_SIZE

  # -------------------------------------------------
  if save:
    ## Raster plots of firings
    fig1 = plt.figure()
    if firings1.size != 0:
      plt.subplot(211)
      plt.scatter(firings1[:, 0], firings1[:, 1] + 1, marker='.')
      plt.xlim(0, T)
      plt.ylabel('Neuron number')
      plt.ylim(0, NUM_EXCITORY+1)
      plt.title('Population 1 firings for p =' + str(p))
  
    if firings2.size != 0:
      plt.subplot(212)
      plt.scatter(firings2[:, 0], firings2[:, 1] + 1, marker='.')
      plt.xlim(0, T)
      plt.ylabel('Neuron number')
      plt.ylim(0, NUM_INHIBITORY+1)
      plt.xlabel('Time (ms)')
      plt.title('Population 2 firings for p =' + str(p))
  
    
    path = os.path.join(DIR_PATH, 'firings_'+str(p)+'.svg') # file name and path
    fig1.savefig(path)
  
    ## Mean firing rate
    fig2 = plt.figure()
    if firings1.size != 0:
      plt.plot(mean_time, mean_firings)
      plt.ylabel('Mean firing rate')
      plt.title('Mean firing rate for p =' + str(p))
  
    path = os.path.join(DIR_PATH, 'mean_firing_'+str(p)+'.svg') # file name and path
    fig2.savefig(path)
  # -------------------------------------------------
  return mean_firings, p

def simulation_wrapper_star(T_p):
    return simulation_wrapper(*T_p)
