import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from IzNetwork import IzNetwork


def CompareMatrix(MatA,MatB,NCol,NRow):
  for i in range(NRow):
    for j in range(NCol):
      if MatA[i,j] != MatB[i,j]:
        print i, j

def CropMatrix(CIJ, StartX, EndX, StartY, EndY):
  NewCIJ = np.zeros([EndX - StartX, EndY - StartY])
  for i in range(StartX, EndX):
    new_i = i - StartX
    for j in range(StartY, EndY):
      NewCIJ[new_i, j - StartY] = CIJ[i, j]
  return(NewCIJ)

def RewireModularNetwork(CIJ, N, Nm, p):
  """
  Intra-modular connection -> Inter-modular connection
  CIJ = connectivity matrix
  N = number of excitory neuron
  Nm = number of excitory neuron per module
  p = probability of rewiring
  """
  for i in range(N):
    module = i/Nm
    start = module * Nm
    end = (module+1) * Nm
    for j in range(start, end):
        if(CIJ[j,i] and (rn.random() < p)):
            CIJ[j,i] = 0
            k = rn.randint(N)
            while(k/Nm == module or CIJ[k,i] == 1):
                k = rn.randint(N)
            CIJ[k,i] = 1
  return CIJ
    
def IzhikevichModularNetwork(N, K, Nm, Nc, NI):
  """
  Set up small world modular network with N nodes, K modules,
  Nm excitory neuron per module, Nc connections per module,
  NI inhibitory neurons
  """
  CIJ = np.zeros([N, N])
  num_hidden_module = NI/K
  num_excitory_neuron = Nm*K

  for i in range(K):
    # Set up excitory-to-excitory connection per module
    for j in range(Nc):
      [source, target] = rn.randint(Nm, size=2)
      sourceNode = (i * Nm) + source
      targetNode = (i * Nm) + target
      while (CIJ[targetNode, sourceNode] == 1) or (source == target):
        #Repeat until no connection is found between sourceNode and targetNode
        [source, target] = rn.randint(Nm, size=2)
        sourceNode = (i * Nm) + source
        targetNode = (i * Nm) + target
      CIJ[targetNode, sourceNode] = 1
      
  # Set up excitory-to-inhibitory connection
  for i in range(num_excitory_neuron):
    inhib = i/4
    CIJ[num_excitory_neuron+inhib, i] = 1
     
  # Set up outgoing inhibitory-to-excitory connection
  for i in range (Nm*K, N):
      CIJ[:, i] = 1
      #CIJ[i, i] = 0
  
  return(CIJ)

def ConnectIzhikevichNetworkLayers(CIJ, NExcitoryLayer, NInhibitoryLayer):
  Dmax = 20 # Maximum Delay
  network = IzNetwork([NExcitoryLayer, NInhibitoryLayer], Dmax)

  NTotalNeurons = NExcitoryLayer + NInhibitoryLayer

  # Set neuron parameters for excitory layer
  rand = rn.rand(NExcitoryLayer)
  network.layer[0].N = NExcitoryLayer
  network.layer[0].a = 0.02 * np.ones(NExcitoryLayer)
  network.layer[0].b = 0.20 * np.ones(NExcitoryLayer)
  network.layer[0].c = -65 + 15*(rand**2)
  network.layer[0].d = 8 - 6*(rand**2)
  
  ## Factor and delay
  network.layer[0].factor[0] = 17
  network.layer[0].factor[1] = 2
  network.layer[0].delay[0] = rn.randint(1,21,size=[NExcitoryLayer,NExcitoryLayer])
  network.layer[0].delay[1] = np.ones([NExcitoryLayer, NInhibitoryLayer])
 
  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i
  # excitory-to-excitory synaptic weights
  network.layer[0].S[0] = CropMatrix(CIJ, 0, NExcitoryLayer, 0, NExcitoryLayer)
  # inhibtory-to-excitory synaptic weights
  network.layer[0].S[1] = CropMatrix(CIJ, 0, NExcitoryLayer, NExcitoryLayer, NTotalNeurons)
  # inhibtory-to-excitory weights
  rand_array = -1 * rn.random(NInhibitoryLayer*NExcitoryLayer).reshape(NExcitoryLayer,NInhibitoryLayer)
  network.layer[0].S[1] = np.multiply(network.layer[0].S[1],rand_array)

  # Set neuron parameters for inhibitory layer
  rand = rn.rand(NInhibitoryLayer)*0
  network.layer[1].N = NInhibitoryLayer
  network.layer[1].a = 0.02 + 0.08*rand
  network.layer[1].b = 0.25 - 0.05*rand
  network.layer[1].c = -65 * np.ones(NInhibitoryLayer)
  network.layer[1].d = 2 * np.ones(NInhibitoryLayer)
  
  ## Factor and delay
  network.layer[1].factor[0] = 50
  network.layer[1].factor[1] = 1
  network.layer[1].delay[0] = np.ones([NInhibitoryLayer, NExcitoryLayer])
  network.layer[1].delay[1] = np.ones([NInhibitoryLayer, NInhibitoryLayer])
 
  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i
  # excitory-to-inhibtory synaptic weights
  network.layer[1].S[0] = CropMatrix(CIJ, NExcitoryLayer, NTotalNeurons, 0, NExcitoryLayer)
  # inhibtory-to-excitory synaptic weights
  network.layer[1].S[1] = CropMatrix(CIJ, NExcitoryLayer, NTotalNeurons, NExcitoryLayer, NTotalNeurons)
  # excitory-to-inhibtory weights
  rand_array = rn.random(NInhibitoryLayer*NExcitoryLayer).reshape(NInhibitoryLayer,NExcitoryLayer)
  network.layer[1].S[0] = np.multiply(network.layer[1].S[0],rand_array)

  # inhibtory-to-inhibtory weights
  rand_array = -1 * rn.random(NInhibitoryLayer*NInhibitoryLayer).reshape(NInhibitoryLayer,NInhibitoryLayer)
  network.layer[1].S[1] = np.multiply(network.layer[1].S[1],rand_array)
  return(network)
