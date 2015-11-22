import numpy as np
import numpy.random as rn
from Izhikevic import IzhikevichModularNetwork, RewireModularNetwork, ConnectIzhikevichNetworkLayers,CompareMatrix
import matplotlib.pyplot as plt
from jpype import *
import atexit
import os

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

#Start JVM
startJVM(getDefaultJVMPath(), "-Djava.class.path=" + "infodynamics.jar")
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
teCalc = teCalcClass()
atexit.register(shutdownJVM)    #doesn't catch system error

res = []

for i in range(1):
    T  = 1000 * 60 # Simulation time
    Ib = 15    # Base current
    p = 0.01*i  # Rewiring probility
    
    # # override for testing 
    #T = 1000 * 2 # for testing
    #p = 0 # for testing
    
    print str(i) + 'th iteration: p is now: ' + str(p)
    
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
    
    #JIDT
    teCalc.initialise(8)
    teCalc.startAddObservations()
    teCalc.addObservations(mean_firings)
    teCalc.finaliseAddObservations()
    result = teCalc.computeAverageLocalOfObservations()
    res.append([p, result])

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
path = os.path.join(DIR_PATH, 'firings.svg')
fig1.savefig(path)

## Mean firing rate
fig2 = plt.figure()
if firings1.size != 0:
    plt.plot(mean_time, mean_firings)
    plt.ylabel('Mean firing rate')
    plt.title('Mean firing rate')
path = os.path.join(DIR_PATH, 'mean_firing.svg')
fig2.savefig(path)

## Multi-information/Integration
I = np.array(res)
fig3 = plt.figure()
if len(I) != 0:
    plt.scatter(I[:, 0], I[:, 1], marker='.')
    plt.ylim(0, 5)
    plt.ylabel('Integration(bits)')
    plt.xlim(0, 1)
    plt.xlabel('Rewiring probability p')
    plt.title('Integration')
path = os.path.join(DIR_PATH, 'integration.svg')
fig3.savefig(path)

plt.show()
shutdownJVM()