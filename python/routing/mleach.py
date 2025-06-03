import csv
import logging
import pickle
import sys
from datetime import datetime
from itertools import islice

import antropy as ant
import numpy as np
import pandas as pd
import skfuzzy
#from pyentrp import entropy as ent
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.special import entr
from skfuzzy import cluster, membership

import config as cf
from python.network.flow_table import FlowTable
from python.network.network import Network
from python.network.node import *
from python.network.sleepscheduler import SleepScheduler
from python.routing.mte import *
from python.routing.routing_protocol import *
#from python.sleep_scheduling.sleep_scheduler import *
from python.utils.markov_model import MarkovChain
from python.utils.utils import get_result_path

model_in = open('C:/Users/PRAG/Desktop/IMPLEMENTATIONS/sdwsn-new-arima/RF_model.pickle','rb')
classifier = pickle.load(model_in)
"""Every node communicate its position to the base station. Then the  Controllers/ BS uses Fuzzy C-Means
Clustering to define clusters and broadcast this information to the network. Finally, a round is executed.

"""
def get_no_of_regular_neighbours(node, network):
    no_  = 0
    for other_node in network.get_alive_nodes():
        if isinstance(other_node,Controller) == False:
            if node != other_node:
                if node.membership == other_node.membership:
                    if other_node.MR==1:
                        no_ += 1
    return no_    

def get_no_of_alive_neighbours(node, network):
    no_  = 0
    for other_node in network.get_alive_nodes():
        if isinstance(other_node,Controller) == False:
            if node != other_node:
                if node.membership == other_node.membership:
                   no_ += 1
    return no_  

def standard_deviation(row, center):
    '''
    Function:
        Calculate the standard deviation of the elements in the row around the center

    Parameters:
        row: A list with node residual energy after every round,
        center: The deviation is calculated around the center, in this case the center is the initial energy; 2 Joules
    '''
    deviation = 0
    for energy in row:
        deviation = deviation + (energy - center)**2
    standard_deviation = (deviation / len(row))**0.5
    return standard_deviation

class MLC(RoutingProtocol):
    """ Modified Leach with Controller Nodes """
    itrr  = 0
    this_remaining_energies_result_path = ''
    # Controller network sizes
    width = cf.AREA_WIDTH
    length = cf.AREA_LENGTH/cf.NB_CONTROLLERS

    networks = [] # store the controller networks
    no_of_times_am_head ={}
    for i in range(0, cf.NB_NODES):
        no_of_times_am_head['id '+str(i)] = 0
    clusters = {}

    def ch_selection_energy_function(self, node, subc):
        """ An energy function to compute the optimal energy of node to be Cluster head
        taking into account the distance of node from base station and energy level. The
        higher the value, the more likely the node is best fit to be cluster head
        """
        distance_from_subc = calculate_distance(node, subc)
        edge_node = Node(0)
        edge_node.pos_x = self.width/2
        edge_node.pos_y = 0

        max_possible_distance = calculate_distance(edge_node, subc)
        return 0.1*(max_possible_distance - distance_from_subc) + 0.9*(node.energy_source.energy)

    def get_CH_neighbour(self, nodez, network=None):
        highest_energy = cf.MINUS_INFINITY
        energies_list = []
        global next_head2
        global sorted_list
        for node in network:
            if node != nodez:
                energy = node.energy_source.energy
                energies_list.append(energy)
                sorted_list = sorted(energies_list)
        for other_node in network:
            if other_node.energy_source.energy == sorted_list[-1]:
                next_head2 = other_node
        return next_head2 


    def node_in_controller_region(self, node, controller):
        """ Returns the controller in the region of the node """
        return (controller.pos_x - MLC.length/2) <= node.pos_x and \
            (MLC.length/2 + controller.pos_x) > node.pos_x

    def pre_communication(self, network):
        """This method is called before round 0."""
        super().pre_communication(network)
        centroids = []

        
        for controller in network.controller_list:
            nodes = [node for node in network.get_sensor_nodes() if \
                        self.node_in_controller_region(node, controller)]      
            controller_network = Network(sensor_nodes= nodes)
            MLC.networks.append(controller_network)
        
        nodeidlist = []
        for node in MLC.networks[0]:
            nodeidlist.append(node.id)

        nodeidlist = []
        for node in MLC.networks[1]:
            nodeidlist.append(node.id)
        # Initialize Energy Maps for each node
        for node in network.get_sensor_nodes():
            if isinstance(node,Controller) == False:
                node.energy_map.append(node.energy_source.energy)

    def setup_phase(self, network, round_nb):
        """ Network Setup Phase. The network setup for all the controller networks
        are executed
        """
        for i, ntwk in enumerate(MLC.networks):
            self._setup_phase_network(ntwk, round_nb, i)
            controller = ntwk[-1]
            controller.transmit(destination=network.get_BS())
        # Update Energy Map at the end of each round
        for node in network.get_sensor_nodes():
            if isinstance(node,Controller) == False:
                node.energy_map.append(node.energy_source.energy)
                node.trackker_from_markov_interval += 1

        round_energies = []

        if round_nb == 0:
            #Perform prediction operations
            this_remaining_energies_result_dir = os.path.join(get_result_path(), 'markov_predictions')
            if not os.path.exists(this_remaining_energies_result_dir):
                os.makedirs(this_remaining_energies_result_dir)
            this_remaining_energies_result_path = os.path.join(this_remaining_energies_result_dir, 'MLC_' + datetime.today().strftime('%H-%M-%S') + '_markov_predictions.csv')
            cf.markov_path = this_remaining_energies_result_path
            round_energies_result_csv =  open(this_remaining_energies_result_path, mode='w')
            round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = [f'node_{i}' for i in range(cf.NB_NODES)]
            round_energies_result_csv_writer.writerow(header)
            round_energies_result_csv.close()
        
        if round_nb > 0:
            if round_nb % cf.MARKOV_PREDICTION_INTERVAL == 0:              
                for node in network.get_sensor_nodes():
                  node.trackker_from_markov_interval = 0
                  if isinstance(node,Controller) == False:
                    transition_matrix = node.generate_transition_matrix()  
                    markov_model = MarkovChain()
                    try:
                        predictions = markov_model.predict(node.transitions[-1], transition_matrix, no_predictions=cf.MARKOV_PREDICTION_INTERVAL)
                    except:
                         predictions = np.zeros(cf.MARKOV_PREDICTION_INTERVAL)
                    amn = pd.DataFrame(predictions)
                    amnn = (amn.values).flatten()
                    amnn1 = amnn.tolist() 
                    amnn1.insert(0, node.energy_source.energy)
                    a=0
                    My_list = [amnn1[0]-amnn1[1]]
                    for x in amnn1[2:]:
                            x1 = My_list[a]-x
                            My_list.append(x1)
                            a=a+1
                    predictions = My_list
                    node.predicted_energy_consumed = predictions
                    node.predicted_remain_energy_list = []
                    initial_remaining = node.energy_source.energy
                    predicted_remaining_energy = np.zeros(cf.MARKOV_PREDICTION_INTERVAL)
                    predicted_remaining_energy[0] =  initial_remaining - node.predicted_energy_consumed[0]
                    predicted_remaining_energy[0] = 0 if predicted_remaining_energy[0] < 0 else predicted_remaining_energy[0]
                    for i in range(1,cf.MARKOV_PREDICTION_INTERVAL):
                        predicted_remaining_energy[i] = predicted_remaining_energy[i-1] -node.predicted_energy_consumed[i]
                        if predicted_remaining_energy[i] < 0:
                            predicted_remaining_energy[i] = 0
                        initial_remaining -= node.predicted_energy_consumed[i]
                    node.predicted_remain_energy_list = predicted_remaining_energy
                    amn = pd.DataFrame(predicted_remaining_energy)
                    amnn = (amn.values).flatten()
                    amnn1 = amnn.tolist()
                    predicted_remaining_energy.flatten()
                    round_energies.append(predicted_remaining_energy)
                    deviation_from_initial_energy = standard_deviation(node.predicted_energy_consumed, node.energy_source.energy)
                    entropy = entr([eel for eel in predictions]).sum()
                    entropy = 0 if entropy == -(np.inf) or entropy == np.nan else entropy
                    if (entropy < -9.816723065 or deviation_from_initial_energy < 2.07E-05):
                        node.MR = 1 
                    else:
                        node.MR = 0
                    node.MR = classifier.predict(np.array([entropy, deviation_from_initial_energy]).reshape(1, -1))
                    if isinstance(node, Node):
                        node.Q = (cf.ALPHA * get_no_of_alive_neighbours(node, network)) + (cf.BETA * get_no_of_regular_neighbours(node, network))
                        if node.Q <= cf.Q_min:
                            node.DC = cf.DC_min
                        if node.Q > cf.Q_min and node.Q  < cf.Q_max:
                            node.DC = cf.DC_med
                        if node.Q >= cf.Q_max:
                            node.DC = cf.DC_max

                cf.ITTR_COUNT += 1
                round_energies_result_csv =  open(cf.markov_path, mode='a')
                round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
                round_energies_result_csv_writer.writerow(round_energies)
                round_energies_result_csv.close()
                if cf.ITTR_COUNT>=2:
                    cf.ITTR_COUNT=0 

    def _setup_phase_network(self, network, round_nb, controller):
        """The base station/controller uses Fuzzy C-Means to clusterize the network. The
        optimal number of clusters is calculated.
        The Cluster heads are determine by the base station using the `Algorithm` based on
        Paper X for each cluster (only in the initial round).
        Then each cluster head chooses a new cluster head for the next round.
        """
        logging.debug('MLC: %s -> setup phase', network.id)
        if round_nb == 0:
            self.clusterize_network(network)
        else:
            
            self.head_rotation(network, round_nb)


    def head_rotation(self, network, round_nb):
        """ Current cluster heads choose next cluster head based on `Algorithm` """

        logging.debug('MLC: Cluster Head Rotation')
        sensor_nodes = [node for node in network if node.energy_source.energy != 0]
        def transform(node): return calculate_distance(node, network[-1])
        distances_to_SUBC = [transform(node) for node in sensor_nodes]
        avg_distance_to_BS = np.average(distances_to_SUBC)
        nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))
        count = 0
        count1=0
        highest_energy = cf.MINUS_INFINITY
        for cluster_id in range(0, nb_clusters):
            cluster = network.get_nodes_by_membership(cluster_id)
            cluster_for_CH = cluster
            for node in cluster:
                if node.next_hop == cf.SUBCONT0 or node.next_hop == cf.SUBCONT1:
                    if node.energy_source.energy <= 1:
                        count+=1
            if len(cluster) <= self.clusters[cluster_id]*0.75:
                    count1+=1
            if count1 >= 0.1*nb_clusters:
                return self.clusterize_network(sensor_nodes)
            else:
                    if len(cluster) == 0:
                            continue        
                    if count >= 0.3*nb_clusters:
                        for node in cluster:
                            if node.next_hop == cf.SUBCONT0:
                                self.current_head = node
                                node.cluster_head_times.append(True) # append True to the list of the node has been a cluster head
                            elif node.next_hop == cf.SUBCONT1:
                                self.current_head = node
                                node.cluster_head_times.append(True)  # append True to the list of the node has been a cluster head
                            else:
                                node.cluster_head_times = []
                            
                        for node in cluster:
                            if node.next_hop == cf.SUBCONT0 or node.next_hop == cf.SUBCONT1:
                                if node.energy_source.energy <= 0.5:
                                    if len(cluster) > self.clusters[cluster_id]*0.2:
                                        cluster_for_CH.remove(node)
                                        node.cluster_head_times = []
                        for node in cluster_for_CH:
                            node_energy = self.ch_selection_energy_function(node, network[-1])
                            if node_energy > highest_energy:
                                    highest_energy = node_energy
                                    next_head = node
                                    next_head2 = self.get_CH_neighbour(next_head, network=cluster)

                            if len(cluster)>10:
                                split_list = [round(0.6 * len(cluster)), round(0.4 * len(cluster))]
                                temp = iter(cluster)
                                res = [list(islice(temp, 0, ele)) for ele in split_list]
                                for node in cluster:
                                    if node in res[0][0:]:
                                        node.next_hop = next_head.id    
                                        next_head.next_hop = network[-1].id
                                    else:
                                        node.next_hop = next_head2.id
                                        next_head2.next_hop = network[-1].id
                            else:
                                node.next_hop = next_head.id

                            if len(sensor_nodes[:-1]) == 1:
                                next_head = sensor_nodes[0]
                                next_head.next_hop = network[-1].id
                            count = 0

    def clusterize_network(self, network):
        """ Create clusters from the network using FCM """

        logging.debug("MLC:: Reconstituting Cluster heads")
        sensor_nodes = [node for node in network if node.energy_source.energy != 0]
        def transform(node): return calculate_distance(node, network[-1])
        distances_to_SUBC = [transform(node) for node in sensor_nodes]
        avg_distance_to_BS = np.average(distances_to_SUBC)
        nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))
        cf.NB_CLUSTERS = nb_clusters
        data = [[node.pos_x, node.pos_y]
                for node in sensor_nodes]
        if not data:
            return
        data = np.array(data).transpose()
        centroids, membership = skfuzzy.cluster.cmeans(data, nb_clusters,
                                                       cf.FUZZY_M, error=0.005,
                                                       maxiter=1000,
                                                       init=None)[0:2]
        membership = np.argmax(membership, axis=0)
        heads = []
        for cluster_id, centroid in enumerate(centroids):
            tmp_centroid = Node(0)
            tmp_centroid.pos_x = centroid[0]
            tmp_centroid.pos_y = centroid[1]
            self.clusters[cluster_id] = 0

            nearest_node = None
            max_energy_function= cf.MINUS_INFINITY
            for node in sensor_nodes:
                if node in heads:
                    continue
                energy_function = self.ch_selection_energy_function(node, network[-1])
                if energy_function > max_energy_function:
                    nearest_node = node
                    max_energy_function = energy_function
            if nearest_node:
                nearest_node.next_hop = network[-1].id
                nearest_node.membership = cluster_id
                heads.append(nearest_node)
        for i, node in enumerate(sensor_nodes):
            if node in heads:
                continue
            if isinstance(node,Controller):
                continue
            cluster_id = membership[i]
            node.membership = cluster_id
            head = [x for x in heads if x.membership == cluster_id][0]
            head2 = self.get_CH_neighbour(head, network[0:-1])
            
            if len(sensor_nodes)>10:
                split_list = [round(0.6 * len(sensor_nodes)), round(0.4 * len(sensor_nodes))]
                temp = iter(sensor_nodes)
                res = [list(islice(temp, 0, ele)) for ele in split_list]
                if node in res[0][0:]:
                    node.next_hop = head.id    
                    head.next_hop = network[-1].id
                else:
                    node.next_hop = head2.id
                    head2.next_hop = network[-1].id
            else:
                node.next_hop = head.id
            self.clusters[cluster_id] += 0

    def plot_network(self, round_nb):
        """ Plots the network topology """
        for i, ntwk in enumerate(MLC.networks):
            plot_clusters(ntwk, save=True, filename=f"{get_result_path()}controller-{i}--round-{round_nb}-network-structure.png")

    def export_energy_map(self, round_nb):
        """ Plots the network topology """
        for i, ntwk in enumerate(MLC.networks):
            plot_clusters(ntwk, save=True, filename=f"{get_result_path()}controller-{i}--round-{round_nb}-network-structure.png")

    def calculate_average_energy(self, nodes):
        x = np.array([node.energy_source.energy for node in nodes])
        return x.mean()

