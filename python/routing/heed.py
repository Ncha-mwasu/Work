import numpy as np
import copy as cp

# def HEED_algo(R, D, p, pMin, E, Emax, net, cost):
#     N = net.shape[1]  # number of nodes
#     CH_prop = max([p * (E / Emax), pMin * np.ones((1, N))])
#     CH_index = np.where(p * (E / Emax) >= CH_prop)[0]
#     CH = np.zeros((1, N))
#     CH[0, CH_index] = 1
#     for i in range(N):
#         if CH[0, i] == 1:
#             net[2, i] = i
#         else:
#             min_cost = min(cost[CH == 1])
#             net[2, i] = np.where(cost == min_cost)[0][0]
#     CH_F = np.where(CH == 1)[1]
#     return CH_F, net



def setup_phase(network):
    
    #initilization
    MAX_ENERGY = 2 # Maximum energy per node
    P_MIN = 1/2
    C_PROB = 0.05
    
    sensor_nodes = network.get_sensor_nodes()
    sensor_node_energy = sorted(sensor_nodes, key=lambda node: ((node.energy_source.energy/MAX_ENERGY) * C_PROB))
    sensor_node_energy.append(P_MIN)
    sensor_node_energy.sort()
    CH_prob = sensor_node_energy[-1]\
    
    
    # C_PROB * len(sor)
    
    # Loop Stage:
    
        
    
    
    
    
    
    
    
    
    
    
    
        
        
    