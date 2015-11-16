import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from IzNetwork import IzNetwork


def CompareMatrix(MatA,MatB,NCol,NRow):
  for i in range(NRow):
    for j in range(NCol):
      if MatA[i,j] != MatB[i,j]:
        print "WTF"

def CropMatrix(CIJ, StartX, EndX, StartY, EndY):
  NewCIJ = np.zeros([EndX - StartX, EndY - StartY])
  for i in range(StartX, EndX):
    new_i = i - StartX
    for j in range(StartY, EndY):
      NewCIJ[new_i, j - StartY] = CIJ[i, j]
  return(NewCIJ)

def FlipMatrix(CIJ, X, Y):
  NewCIJ = np.zeros([Y, X])
  for i in range(X):
    for j in range(Y):
      NewCIJ[j, i] = CIJ[i, j]
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
    module = (i / Nm)
    start = module * Nm   #start of module
    end = start + Nm #end of module
    for j in range(start, end):
      if (CIJ[i,j]) and (rn.random() < p):
        CIJ[i, j] = 0
        h = rn.randint(N)
        while (h / Nm) == module:
          #h needs to be in another module # TODO: does it have to be?
          h = rn.randint(N)
        CIJ[i, h] = 1
  return(CIJ)
    
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
      [source, target] = rn.randint(Nm, size=2) # TODO: this allows self connection correct?
      sourceNode = (i * Nm) + source
      targetNode = (i * Nm) + target
      while (CIJ[sourceNode, targetNode] == 1) or (source == target):
        #Repeat until no connection is found between sourceNode and targetNode
        [source, target] = rn.randint(Nm, size=2) # TODO: this allows self connection correct?
        sourceNode = (i * Nm) + source
        targetNode = (i * Nm) + target
      CIJ[sourceNode, targetNode] = 1
    # Set up excitory-to-inhibitory connection // iterating through all the inhibitory neurons
    for k in range(num_hidden_module):
      # Number of excitory neurons + the i module of inhibitory neurons + k
      targetInhibitory = num_excitory_neuron + (i * (num_hidden_module)) + k
      # Connect 4 excitory neurons to 1 inhibitory neuron # TODO: why is this the one to connect?
      startingSource = (i * Nm) + (k * 4) 
      CIJ[startingSource, targetInhibitory] = 1
      CIJ[startingSource + 1, targetInhibitory] = 1
      CIJ[startingSource + 2, targetInhibitory] = 1
      CIJ[startingSource + 3, targetInhibitory] = 1
      # Set up outgoing inhibitory-to-excitory connection
      for l in range(N):
        CIJ[targetInhibitory, l] = 1
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
  network.layer[0].delay[0] = np.ones([NExcitoryLayer, NExcitoryLayer])

  # for i in range(NExcitoryLayer):
  #   for j in range(NExcitoryLayer):
  #     network.layer[0].delay[0][i,j] = rn.randint(1,20)
  network.layer[0].delay[0] = rn.randint(1,20,size=[NExcitoryLayer,NExcitoryLayer])
  network.layer[0].delay[1] = np.ones([NExcitoryLayer, NInhibitoryLayer])
 
  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i

  # excitory-to-excitory synaptic weights
  network.layer[0].S[0] = FlipMatrix(CropMatrix(CIJ, 0, NExcitoryLayer,
     0, NExcitoryLayer), NExcitoryLayer, NExcitoryLayer)
  # inhibtory-to-excitory synaptic weights
  network.layer[0].S[1] = FlipMatrix(CropMatrix(CIJ, NExcitoryLayer, NTotalNeurons,
     0, NExcitoryLayer), NTotalNeurons -  NExcitoryLayer, NExcitoryLayer) # target neuron->rows, source neuron->columns
  
  # plt.matshow(network.layer[0].S[0], cmap=plt.cm.gray)
  # plt.show()
  # plt.matshow(network.layer[0].S[1], cmap=plt.cm.gray)
  # plt.show()
  
  # inhibtory-to-excitory weights
  # for i in range(NExcitoryLayer):
  #   for j in range(NInhibitoryLayer):
  #     network.layer[0].S[1][i,j] = network.layer[0].S[1][i,j] * rn.random() * -1
  rand_array = -1 * rn.random(NExcitoryLayer*NInhibitoryLayer).reshape(NExcitoryLayer,NInhibitoryLayer)
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
  network.layer[1].S[0] = FlipMatrix(CropMatrix(CIJ, 0, NExcitoryLayer,
     NExcitoryLayer, NTotalNeurons), NExcitoryLayer, NTotalNeurons - NExcitoryLayer)
  # inhibtory-to-excitory synaptic weights
  network.layer[1].S[1] = FlipMatrix(CropMatrix(CIJ, NExcitoryLayer, NTotalNeurons,
     NExcitoryLayer, NTotalNeurons), NTotalNeurons - NExcitoryLayer, NTotalNeurons - NExcitoryLayer)
  
  # plt.matshow(network.layer[1].S[0], cmap=plt.cm.gray)
  # plt.show()
  # plt.matshow(network.layer[1].S[1], cmap=plt.cm.gray)
  # plt.show()
  # excitory-to-inhibtory weights
  # for i in range(NInhibitoryLayer):
  #   for j in range(NExcitoryLayer):
  #     network.layer[1].S[0][i,j] = network.layer[1].S[0][i,j] * rn.random()
  rand_array = rn.random(NExcitoryLayer*NInhibitoryLayer).reshape(NInhibitoryLayer,NExcitoryLayer)
  network.layer[1].S[0] = np.multiply(network.layer[1].S[0],rand_array)



  # inhibtory-to-inhibtory weights
  # for i in range(NInhibitoryLayer):
  #   for j in range(NInhibitoryLayer):
  #     network.layer[1].S[1][i,j] = network.layer[1].S[1][i,j] * rn.random() * -1
  rand_array = -1 * rn.random(NInhibitoryLayer*NInhibitoryLayer).reshape(NInhibitoryLayer,NInhibitoryLayer)
  network.layer[1].S[1] = np.multiply(network.layer[1].S[1],rand_array)
  return(network)
