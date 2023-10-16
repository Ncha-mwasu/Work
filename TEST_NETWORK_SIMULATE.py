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

def f():
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
    
    markov_intervals = [ 1,  1]
    time_steps = [ 1.5e-9, 1.5e-4]
    c = list(itertools.product(markov_intervals, time_steps))
     # to make a cross product ofthe two matrices to pick all possible pairs of markov_intervals and time steps 
     #  in the form [(2, 1.5e-09), (2, 0.00015), (5, 1.5e-09), (5, 0.00015)]


    
 
    j = []
    for i in range(5):
        # if i == 4:
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

        cf.MARKOV_PREDICTION_INTERVAL = 3
        cf.time_steps = 1.5e-4
        j .append((network.simulate('MLC')))
        print('error+mse ', j) # i have modified network.simluate to return the mean squared error+mean markov energy, refer to line 228 in network.py
    return np.array(j)


f()
print('finished')
