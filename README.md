# Computational Neurodynamics Coursework (Kuohern Wong, Ryan Tam, Cherie Pun)

## Question 1
To generate the graphs,  run ```python Q1.py```. The script saves the following graphs in the directory q1/p*x*/, where *x* stands for the probability used in the simulation.
- Connectivity matrix of the excitory neurons (connectivity_matrix_*x*.svg)
- The firings scatter graph of excitory and inhibitory neuron layer (firings_*x*.svg)
- The mean firing rate of the 8 modules of excitory neurons (mean_firing_*x*.svg)

Plots for *p* = {0, 0.1, 0.2, 0.3, 0.4, 0.5} can be found in [q1_example/](https://github.com/khw12/brainAndStuff/tree/master/q1_example).

## Question 2
To generate the graphs,  run ```python Q2.py```. The script saves the following graph in the directory q2/
- The multi-information plot for *n* number of trials (integration_*n*.svg)

Plots for *n* = {20, 50, 500} can be found in [q2_example/](https://github.com/khw12/brainAndStuff/tree/master/q2_example).

## Remark ##
### Parallelisation
Simulation of the modular networks was wrapped so parallelisation can be easily achieved using `multiprocessing` with the help of `itertools`. In both `Q1.py` and `Q2.py`, we have added `try` and `except` statements to proceed with the simulation with 1 core only should any exception occur (eg. failed to import multiprocessing module). 

### Izhikevich.py and Run.py
Izhikevich.py contains helper functions that is used in Q1 and Q2. `GenerateNetwork` creates an izhikevich neuron network. `CompareMatrix` compares 2 matrices and help in debugging differences in matrices. `IzhikevichModularNetwork` generates the connectivity matrix for the izhikevich network.Run.py contains the functions needed to perform the simulation on the given network. `RunSimulation` simulates the network for a given T and introduces background firing as well.
