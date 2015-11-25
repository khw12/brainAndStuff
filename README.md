# Computational Neurodynamics Coursework (Kuohern Wong, Ryan Tam, Cherie Pun)

## Question 1
To generate the graphs,  run ```python Q1.py```. The script saves the following graphs in the directory q1/px, where x stands for the probability used in the simulation.
- Connectivity matrix of the excitory neurons
- Membrane potentials of excitory and inhibitory neuron layer
- Recovery variable of excitory and inhibitory neuron layer
- The firings scatter graph of excitory and inhibitory neuron layer
- The mean firing rate of the 8 modules of excitory neurons


## Question 2
To generate the graphs,  run ```python Q2.py```. The script saves the following graphs in the directory q2/
- The firings scatter graph of excitory and inhibitory neuron layer
- The mean firing rate of the 8 modules of excitory neurons
- The multiinfo graph


## Remark ##
### Parallelisation
Simulation of the modular networks was wrapped so parallelisation can be easily achieved using `multiprocessing` with the help of `itertools`. In both `Q1.py` and `Q2.py`, we have added `try` and `except` statements to proceed with the simulation with 1 core only should any exception occur (eg. failed to import multiprocessing module). 
