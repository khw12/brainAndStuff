import numpy as np
import numpy.random as rn
from Izhikevic import IzhikevichModularNetwork, RewireModularNetwork, CropMatrix, ConnectIzhikevichNetworkLayers,CompareMatrix
import matplotlib.pyplot as plt
import copy
import os
    
NUM_NEURONS = 1000
NUM_MODULES = 8
NUM_EXCITORY = 800
NUM_EXCITORY_PER_MODULE = 100
NUM_INHIBITORY = 200
NUM_CONNECTIONS_E_to_E = 1000
NUM_CONNECTIONS_E_to_I = 4

T  = 1000  # Simulation time
Ib = 15    # Base current
# p = 0      # Rewiring probility

def run_simulation(p):
	"""
	Run a simulation with p as rewiring probability  
	"""

	CIJ = IzhikevichModularNetwork(NUM_NEURONS, NUM_MODULES, NUM_EXCITORY_PER_MODULE, NUM_CONNECTIONS_E_to_E, NUM_INHIBITORY)
	CIJ = RewireModularNetwork(CIJ, NUM_EXCITORY, NUM_EXCITORY_PER_MODULE, p)
	net = ConnectIzhikevichNetworkLayers(CIJ, NUM_EXCITORY, NUM_INHIBITORY)
	## Initialise layers
	for lr in xrange(len(net.layer)):
	    net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
	    net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
	    net.layer[lr].firings = np.array([])

	v1 = np.zeros([T, NUM_EXCITORY])
	v2 = np.zeros([T, NUM_INHIBITORY])
	u1 = np.zeros([T, NUM_EXCITORY])
	u2 = np.zeros([T, NUM_INHIBITORY])

	## SIMULATE
	for t in xrange(T):
	    net.layer[0].I = np.zeros(NUM_EXCITORY)
	    net.layer[1].I = np.zeros(NUM_INHIBITORY)
	    
	    # Background firing
	    for i in range(NUM_EXCITORY):
	        if np.random.poisson(0.01) > 0:
	            net.layer[0].I[i] = Ib
	        
	    for i in range(NUM_INHIBITORY):
	        if np.random.poisson(0.01) > 0:
	            net.layer[1].I[i] = Ib
	            
	    net.Update(t)

	    v1[t] = net.layer[0].v
	    v2[t] = net.layer[1].v
	    u1[t] = net.layer[0].u
	    u2[t] = net.layer[1].u

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
	mean_firings = np.zeros([50,8])
	mean_time = range(0,1000,20)

	# note firings is array of array of [t f] where t is timestamp and f is source 
	for [idt,fired] in firings1:
		for window, t_start in enumerate(mean_time):
			if (t_start + 50 > idt) & (t_start <= idt):
				# recover the module from nueron number
				module = fired/NUM_EXCITORY_PER_MODULE
				# no need to filter inhib since its firing1
				mean_firings[window,module] += 1

	mean_firings /= 50

	## Raster plots of firings
	figure3 = plt.figure(3)
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
	figure4 = plt.figure(4)
	if firings1.size != 0:
	    plt.plot(range(0, 1000, 20), mean_firings)
	    plt.ylabel('Mean firing rate')
	    plt.title('Mean firing rate')

	plt.show()


from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool(3)
    p.map(run_simulation, [0,0.1,0.2])





