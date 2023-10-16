'''
I create a PSO with 2 variables (time step and markov interval)
with the cbjective to minimize... <<energy + error>>

'''
# Import modules
import numpy as np
import config as cf
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
#from pymoo.constraints.as_penalty import ConstraintsAsPenalty
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}



import inspect
import uuid
import os
import argparse
import logging
import sys
from collections import OrderedDict
import matplotlib
# For CI-Environments run matplotlib without any display backend
if os.environ.get("DRY_RUN"):
    matplotlib.use('Agg')

import config as cf
from python.network.aggregation_model import *
from python.network.network import Network
from python.routing.direct_communication import *
from python.routing.fcm import *
from python.routing.leach_backup import *
from python.routing.mleach import *
from python.routing.mte import *
from python.utils.tracer import *
from python.utils.utils import *
from python.network.node import Controller
import math
import numpy as np
import itertools 
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher

markov_intervals = [ 3,  4] # add the intervals you need
time_steps = [ 10e-6, 20e-6, 30e-6, 40e-6] # add the time steps you need
c = list(itertools.product(markov_intervals, time_steps))
once= True
all_particles = []
def f(x):
    """Higher-level method to run an instance
        of the network with certain parameters

        Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed energy + error loss for each particle
    """
    

     # to make a cross product ofthe two matrices to pick all possible pairs of markov_intervals and time steps 
     #  in the form [(2, 1.5e-09), (2, 0.00015), (5, 1.5e-09), (5, 0.00015)]


    
    n_particles = x.shape[0]
    print('x is............',x)
    
    if  cf.once:
        global all_particles;
        all_particles.append(x)

    
        # cf.once = False
    print('==========', n_particles)
    j = []
    for i in range(n_particles):
        print("INSTANCER ", i)
        subcontr0 = Controller(cf.SUBCONT0, cf.SUB_CON0_POS_X, cf.SUB_CON0_POS_Y)
        subcontr1 = Controller(cf.SUBCONT1, cf.SUB_CON1_POS_X, cf.SUB_CON1_POS_Y)

        network = Network(cont_nodes=[subcontr0, subcontr1])
        routing_topology = 'MLC'
        network.set_scenario('MLC')
        routing_protocol_class = eval(routing_topology)
        network.routing_protocol = routing_protocol_class()

        aggregation_function = 'total' + '_cost_aggregation'
        network.set_aggregation_function(eval(aggregation_function))

        # cf.MARKOV_PREDICTION_INTERVAL = c[i][0]
        # cf.time_steps = c[i][1]

   
        # to use the actual randomly inititalized particles
        cf.MARKOV_PREDICTION_INTERVAL = round(x[i][0])  if round(x[i][0]) >= 1 else round(x[i][0])+1
        cf.time_steps = x[i][1]

        j.append((network.simulate('MLC'))) # i have modified network.simluate to return the mean squared error+mean markov energy, refer to line 228 in network.py
    return np.array(j)




# Initialize swarm
'''
                 * c1 : float
                cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
'''
# options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}

bound = (np.array([0.000000, 0.00000]), np.array([5, 5])) # boundary if using random initialization

dimensions = 2 
'''
number of particles must be set to  =  no of markov intervals * no of time steps
'''

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=8, dimensions=dimensions, options=options, bounds=bound)
# n_particlws = 4 because 2 time steps+ 2 prediction intervals defined in line 60 & 61

# Perform optimization
cost, pos = optimizer.optimize(f, iters=10)


print('------------final bests--------------------')
print('minimized energy + error to=', cost) 

print((np.where(all_particles == pos))[0])

for i, parts in enumerate(all_particles):
    print('parts ',parts)
    print('pos ',pos)
    if True in (parts == pos):
        indextobest = i % len(c)

best_part =c[indextobest]
print('best prediction round interval is ->', best_part[0])
print('best time step is ->', best_part[1])

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()

# Initialize mesher with sphere function
