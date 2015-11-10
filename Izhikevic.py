import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from IzNetwork import IzNetwork

def Run(p):
  CIJ = IzhikevichModularNetwork(1000, 8, 100, 1000, 200)
  CIJ = RewireModularNetwork(CIJ, 800, 100, p)
  plt.matshow(CropMatrix(CIJ, 0, 800, 0, 800), cmap=plt.cm.gray)
  plt.show()
  return(CIJ)

def CropMatrix(CIJ, StartX, EndX, StartY, EndY):
  NewCIJ = np.zeros([EndX - StartX, EndY - StartY])
  for i in range(StartX, EndX):
    for j in range(StartY, EndY):
      NewCIJ[i - StartX, j - StartY] = CIJ[i, j]
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
    start = (i / Nm) * Nm   #start of module
    end = start + Nm #end of module
    for j in range(start, end):
      if CIJ[i,j] and rn.random() < p:
        CIJ[i, j] = 0
        h = rn.randint(N)
        while (h / Nm) == (i / Nm):
          #h needs to be in another module
          h = rn.randint(N)
        CIJ[i, h] = 1
  return(CIJ)
    
def IzhikevichModularNetwork(N, K, Nm, Nc, NI):
  """
  Set up small world modular network with N nodes, k modules,
  Nm excitory neuron per module, Nc connections per module, 
  NI inhibitory neurons
  """
  CIJ = np.zeros([N, N])
  for i in range(K):
    # Set up excitory-to-excitory connection per module
    for j in range(Nc):
      [source, target] = rn.randint(Nm, size=2)
      sourceNode = (i * Nm) + source
      targetNode = (i * Nm) + target 
      while CIJ[sourceNode, targetNode]: 
        #Repeat until no connection is found between sourceNode and targetNode
        [source, target] = rn.randint(Nm, size=2)
        sourceNode = (i * Nm) + source
        targetNode = (i * Nm) + target
      CIJ[sourceNode, targetNode] = 1
    # Set up excitory-to-inhibitory connection // iterating through all the inhibitory neurons
    for k in range(NI/K):
      # Number of excitory neurons + the i module of inhibitory neurons + k
      targetInhibitory = 800 + (i * (NI / K)) + k
      # Connect 4 excitory neurons to 1 inhibitory neuron
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
  Dmax = 25 # Maximum Delay
  network = IzNetwork([NExcitoryLayer, NInhibitoryLayer], Dmax)

  # Set neuron parameters for excitory layer
  rand = rn.rand(NExcitoryLayer)
  network.layer[0].N = NExcitoryLayer
  network.layer[0].a = 0.02 * np.ones(NExcitoryLayer)
  network.layer[0].b = 0.20 * np.ones(NExcitoryLayer)
  network.layer[0].c = -65 + 15*(rand**2)
  network.layer[0].d = 8 - 6*(rand**2)
  network.layer[0].factor[0] = 17
  network.layer[0].factor[1] = 2
  network.layer[0].delay[0] = np.ones([NExcitoryLayer, NExcitoryLayer])
  for i in range(NExcitoryLayer):
    for j in range(NExcitoryLayer):
      network.layer[0].delay[0][i,j] = rn.randint(1,20)
  network.layer[0].factor[1] = np.ones([NExcitoryLayer, NInhibitoryLayer])
  network.layer[0].S[0] = FlipMatrix(CropMatrix(CIJ, 0, 800, 0, 800))
  network.layer[0].S[1] = FlipMatrix(CropMatrix(CIJ, 800, 1000, 0, 800)) # target neuron->rows, source neuron->columns
  for i in range(NExcitoryLayer):
    for j in range(NInhibitoryLayer):
      network.layer[0].S[1][i,j] = network.layer[0].S[1][i,j] * rn.random()

  # Set neuron parameters for inhibitory layer
  rand = rn.rand(NInhibitoryLayer)
  network.layer[1].N = NInhibitoryLayer
  network.layer[1].a = 0.02 * np.ones(NInhibitoryLayer)
  network.layer[1].b = 0.25 * np.ones(NInhibitoryLayer)
  network.layer[1].c = -65 + 15*(rand**2)
  network.layer[1].d = 2 - 6*(rand**2)
  network.layer[1].factor[0] = 50
  network.layer[1].factor[1] = 1
  network.layer[1].factor[0] = np.ones([NInhibitoryLayer, NExcitoryLayer])
  network.layer[1].factor[1] = np.ones([NInhibitoryLayer, NInhibitoryLayer])
  network.layer[1].S[0] = FlipMatrix(CropMatrix(CIJ, 0, 800, 800, 1000))
  network.layer[1].S[1] = FlipMatrix(CropMatrix(CIJ, 800, 1000, 800, 1000))
  for i in range(NInhibitoryLayer):
    for j in range(NExcitoryLayer):
      network.layer[1].S[0][i,j] = network.layer[1].S[0][i,j] * rn.random() * -1
  for i in range(NInhibitoryLayer):
    for j in range(NInhibitoryLayer):
      network.layer[1].S[1][i,j] = network.layer[1].S[1][i,j] * rn.random() * -1

  return(network)