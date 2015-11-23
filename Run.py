import numpy as np

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
      if np.random.poisson(0.01) > 0:
        net.layer[0].I[i] = Ib
        
    #for i in range(NUM_INHIBITORY):
    #  if np.random.poisson(0.01) > 0:
    #    net.layer[1].I[i] = Ib
            
    net.Update(t)

    v1[t] = net.layer[0].v
    v2[t] = net.layer[1].v
    u1[t] = net.layer[0].u
    u2[t] = net.layer[1].u

  return([net, v1, v2, u1, u2])