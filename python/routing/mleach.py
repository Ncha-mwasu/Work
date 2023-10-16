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
    # store the original number of nodes in each cluster. This would be used to determine
    # when the total number of remaining alive nodes in the cluster is less than a certain
    # percentage of the original
    clusters = {}

    def ch_selection_energy_function(self, node, subc):
        """ An energy function to compute the optimal energy of node to be Cluster head
        taking into account the distance of node from base station and energy level. The
        higher the value, the more likely the node is best fit to be cluster head
        """
        distance_from_subc = calculate_distance(node, subc)
        # The distance cost is the difference between the farthest possible distance a node can
        # be from the base station and the distance of the current node from the base station
        edge_node = Node(0)
        edge_node.pos_x = self.width/2
        edge_node.pos_y = 0

        max_possible_distance = calculate_distance(edge_node, subc)
        return 0.1*(max_possible_distance - distance_from_subc) + 0.9*(node.energy_source.energy)
        #return (max_possible_distance - distance_from_subc) * node.energy_source.energy

    def get_CH_neighbour(self, nodez, network=None):
        highest_energy = cf.MINUS_INFINITY
        energies_list = []
        global next_head2
        global sorted_list
        for node in network:
            if node != nodez:
                #ode_energy = self.ch_selection_energy_function(node, network[-1])
                #if node_energy > highest_energy:
                    #highest_energy = node_energy
                    #next_head2 = node
                energy = node.energy_source.energy
                energies_list.append(energy)
                sorted_list = sorted(energies_list)
    #print("sorted", str(sorted_list))
        for other_node in network:
            #for ene in sorted_list[0:-1]:
            if other_node.energy_source.energy == sorted_list[-1]:
                next_head2 = other_node
        return next_head2 


    def node_in_controller_region(self, node, controller):
        """ Returns the controller in the region of the node """
        # If a node is within the controller region
        return (controller.pos_x - MLC.length/2) <= node.pos_x and \
            (MLC.length/2 + controller.pos_x) > node.pos_x

    def pre_communication(self, network):
        """This method is called before round 0."""
        # Setup Controllers on network
        super().pre_communication(network)

        # Nodes closest to centroids i.e
        centroids = []

        
        for controller in network.controller_list:
            #controller.pos_x = MLC.length/2 + (MLC.length*i)
            #controller.pos_y = MLC.width/2
            
            nodes = [node for node in network.get_sensor_nodes() if \
                        self.node_in_controller_region(node, controller)]
            #for node in nodes:
             #   node.membership = controller.id            
            controller_network = Network(sensor_nodes= nodes)
            #controller_network.append(controller)
            MLC.networks.append(controller_network)
        
        nodeidlist = []
        for node in MLC.networks[0]:
            nodeidlist.append(node.id)
            with open('controllernodes1.txt', 'w') as f:
                f.write(str(nodeidlist))

        nodeidlist = []
        for node in MLC.networks[1]:
            nodeidlist.append(node.id)
            with open('controllernodes2.txt', 'w') as f:
                f.write(str(nodeidlist))

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
            # # Controllers send data to Base Station after each round
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
                #print('kaiii')
                for node in network.get_sensor_nodes():
                  node.trackker_from_markov_interval = 0
                  if isinstance(node,Controller) == False:
                    transition_matrix = node.generate_transition_matrix()  
                    markov_model = MarkovChain()
                    #print('node transitions list ->',str(node.transitions))
                    #open('markov_energy.txt', 'a+').write(str(node.transitions.keys))
                    try:
                   # print('Transitions ->', node.transitions)
                        predictions = markov_model.predict(node.transitions[-1], transition_matrix, no_predictions=cf.MARKOV_PREDICTION_INTERVAL)
                    #open('roundseer.txt','a+').write(str(predictions2)+'\n')
                    except:
                         predictions = np.zeros(cf.MARKOV_PREDICTION_INTERVAL)
                    amn = pd.DataFrame(predictions)
                    amnn = (amn.values).flatten()
                    #amnn1 = [node.energy_source.energy - amnn[0], amnn1[0]- amnn[1], amnn1[1]- amnn[2], amnn1[2]- amnn[3], amnn1[3 - amnn[4]]]
        
                    amnn1 = amnn.tolist() 
                    amnn1.insert(0, node.energy_source.energy)
                    #print('-==>', (amnn1))
                    a=0
                    My_list = [amnn1[0]-amnn1[1]]
                    for x in amnn1[2:]:
                            x1 = My_list[a]-x
                            My_list.append(x1)
                            a=a+1
                    #print('->', (My_list))
                    #predictions = [a - b for a, b in self.pairwise(amnn1)]
                    predictions = My_list
                    
                    node.predicted_energy_consumed = predictions
                    # print('predicted energy consumed->',node.predicted_energy_consumed)
                    node.predicted_remain_energy_list = []
                    initial_remaining = node.energy_source.energy
                    predicted_remaining_energy = np.zeros(cf.MARKOV_PREDICTION_INTERVAL)
                    predicted_remaining_energy[0] =  initial_remaining - node.predicted_energy_consumed[0]
                    predicted_remaining_energy[0] = 0 if predicted_remaining_energy[0] < 0 else predicted_remaining_energy[0]
                    # predicted_remaining_energy[0] = initial_remaining - node.predicted_energy_consumed[cf.ITTR_COUNT]
                    for i in range(1,cf.MARKOV_PREDICTION_INTERVAL):
                        predicted_remaining_energy[i] = predicted_remaining_energy[i-1] -node.predicted_energy_consumed[i]
                        if predicted_remaining_energy[i] < 0:
                            predicted_remaining_energy[i] = 0
                        initial_remaining -= node.predicted_energy_consumed[i]
                    node.predicted_remain_energy_list = predicted_remaining_energy
                    amn = pd.DataFrame(predicted_remaining_energy)
                    amnn = (amn.values).flatten()
                    #amnn1 = [node.energy_source.energy - amnn[0], amnn1[0]- amnn[1], amnn1[1]- amnn[2], amnn1[2]- amnn[3], amnn1[3 - amnn[4]]]
        
                    amnn1 = amnn.tolist()
                    predicted_remaining_energy.flatten()
                    # if predicted_remaining_energy < 0:
                    #     predicted_remaining_energy = 0
                    #print('predicted remaining consumed->',predicted_remaining_energy)
                    round_energies.append(predicted_remaining_energy)
                    #deviation_from_initial_energy = cf.INITIAL_ENERGY - standard_deviation(node.predicted_energy_consumed, cf.INITIAL_ENERGY)
                    deviation_from_initial_energy = standard_deviation(node.predicted_energy_consumed, node.energy_source.energy)
                    entropy = entr([eel for eel in predictions]).sum()
                    #std_ts = np.std([predictions])
                    #entropy = ent.sample_entropy([predictions], 4, 0.2 * std_ts).sum()
                    entropy = 0 if entropy == -(np.inf) or entropy == np.nan else entropy
                    if (entropy < -9.816723065 or deviation_from_initial_energy < 2.07E-05):
                        node.MR = 1 
                    else:
                        node.MR = 0
                    #print('trails...',str(entropy))
                    #if node.MR == 0:
                    #open('svm_.txt', 'a+').write("%s %s %s\n" % (str(entropy), str(deviation_from_initial_energy), str(node.MR)))
                    #classes = []
                    #if isinstance(node, Node):
                    node.MR = classifier.predict(np.array([entropy, deviation_from_initial_energy]).reshape(1, -1))
                    #classes = list(node.MR)
                    #with open('SVM_Classes.txt', 'a') as f:
                       # f.write(str(classes) +'\n')
                    # print('predicted energy consumed->',node.predicted_energy_consumed)
                #for node in network.get_alive_nodes():
                    if isinstance(node, Node):
                        node.Q = (cf.ALPHA * get_no_of_alive_neighbours(node, network)) + (cf.BETA * get_no_of_regular_neighbours(node, network))
                        #with open('Q.txt', 'a') as f:
                            #f.write(str(node.Q) +'\n')
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
        

        
        # if round_nb > 0 and round_nb % MARKOV_PREDICTION_INTERVAL == 0:
            
        #     for node in network.get_sensor_nodes():
        #        if isinstance(node,Controller) == False:
        #            predicted_remaining_energy = node.energy_source.energy-node.predicted_energy_consumed[cf.ITTR_COUNT]
        #            if predicted_remaining_energy < 0:
        #                predicted_remaining_energy = 0
        #            print('predicted remaining consumed->',predicted_remaining_energy)
        #            round_energies.append(predicted_remaining_energy)
            #  cf.ITTR_COUNT += 1
           #   round_energies_result_csv =  open(cf.markov_path, mode='a')
          #    round_energies_result_csv_writer = csv.writer(round_energies_result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
         #     round_energies_result_csv_writer.writerow(round_energies)
        #     round_energies_result_csv.close()'''
            

    def _setup_phase_network(self, network, round_nb, controller):
        """The base station/controller uses Fuzzy C-Means to clusterize the network. The
        optimal number of clusters is calculated.
        The Cluster heads are determine by the base station using the `Algorithm` based on
        Paper X for each cluster (only in the initial round).
        Then each cluster head chooses a new cluster head for the next round.
        """

        # Add Controllers to the Nodes if enabled
        logging.debug('MLC: %s -> setup phase', network.id)
        if round_nb == 0:
            # Base station selects clusters and Cluster heads in the first round
            self.clusterize_network(network)
        else:
            # In subsequent rounds, new cluster heads are chosen from cluster util the total
            # number of nodes in the clusters are less than 1/3rd of the original nodes
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
            # cluster = Network(network.get_nodes_by_membership(cluster_id))
            cluster = network.get_nodes_by_membership(cluster_id)
            # When the total number of nodes in a cluster falls below 2/3rd of the original
            # node numbers, reconstitute the clusters
            # print('the clusters are....', cluster)
            # print(len(cluster))
            # print(self.clusters[cluster_id]
            cluster_for_CH = cluster
            for node in cluster:
                if node.next_hop == cf.SUBCONT0 or node.next_hop == cf.SUBCONT1:
                    if node.energy_source.energy <= 1:
                        count+=1
            if len(cluster) <= self.clusters[cluster_id]*0.75:
                    count1+=1
            if count1 >= 0.1*nb_clusters:
                #if count== 3:
                return self.clusterize_network(sensor_nodes)
            else:

            # check if there is someone alive in this cluster
                    if len(cluster) == 0:
                            continue
                    #if round_nb%5 !=0:        
                    if count >= 0.3*nb_clusters:
                        for node in cluster:
                            if node.next_hop == cf.SUBCONT0:
                                self.current_head = node
                                node.cluster_head_times.append(True) # append True to the list of the node has been a cluster head
                            elif node.next_hop == cf.SUBCONT1:
                                self.current_head = node
                    # self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
                                node.cluster_head_times.append(True)  # append True to the list of the node has been a cluster head
                            else:
                                node.cluster_head_times = []
                            
                        for node in cluster:
                            #if len(node.cluster_head_times)>=2:
                            if node.next_hop == cf.SUBCONT0 or node.next_hop == cf.SUBCONT1:
                                if node.energy_source.energy <= 0.5:
                                    if len(cluster) > self.clusters[cluster_id]*0.2:
                                        cluster_for_CH.remove(node)
                                        node.cluster_head_times = []
                        #for node in cluster:
                          #  if node.next_hop == cf.SUBCONT0 or node.next_hop == cf.SUBCONT1:
                            #    if node.energy_source.energy < 1 or node.energy_source.energy == 0:
                        #highest_energy = cf.MINUS_INFINITY
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
                    #print('next heads ', next_head.id)
                                        next_head.next_hop = network[-1].id
            #for node in res[1][0:]:
                                    else:
                                        node.next_hop = next_head2.id
                                        next_head2.next_hop = network[-1].id
                            else:
                                node.next_hop = next_head.id

                            if len(sensor_nodes[:-1]) == 1:
                                next_head = sensor_nodes[0]
                                next_head.next_hop = network[-1].id
                            count = 0



            #print('next hops for heads ', next_head.next_hop)

    def clusterize_network(self, network):
        """ Create clusters from the network using FCM """

        logging.debug("MLC:: Reconstituting Cluster heads")
        sensor_nodes = [node for node in network if node.energy_source.energy != 0]

        #calculate the average distance to the BS
        def transform(node): return calculate_distance(node, network[-1])
        distances_to_SUBC = [transform(node) for node in sensor_nodes]
        avg_distance_to_BS = np.average(distances_to_SUBC)
        nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))
        #nb_clusters = int (0.1* (len(sensor_nodes)))
        #nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))
        #print('len of clusters', nb_clusters)

        cf.NB_CLUSTERS = nb_clusters
        # cf.NB_CLUSTERS = calculate_opt_nb_clusters(len(sensor_nodes,))

        data = [[node.pos_x, node.pos_y]
                for node in sensor_nodes]
        if not data:
            return
        # Use Fuzzy C-Means Clustering to Determine Clusters
        # format data to shape expected by skfuzzy API
        data = np.array(data).transpose()
        centroids, membership = skfuzzy.cluster.cmeans(data, nb_clusters,
                                                       cf.FUZZY_M, error=0.005,
                                                       maxiter=1000,
                                                       init=None)[0:2]
        # print('---', centroids, membership, '------')
        membership = np.argmax(membership, axis=0)
        #print('length of centroid',  len(centroids))

        heads = []
        # also annotates centroids to network
        #network.centroids = []
        for cluster_id, centroid in enumerate(centroids):
            tmp_centroid = Node(0)
            tmp_centroid.pos_x = centroid[0]
            tmp_centroid.pos_y = centroid[1]

            #network.centroids.append(tmp_centroid)
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
                # Nearest Node is made Cluster Head and CH send data to Cluster Controllers
                nearest_node.next_hop = network[-1].id
                nearest_node.membership = cluster_id
                heads.append(nearest_node)

        #for cluster_id in range(0, nb_clusters):
            # cluster = Network(network.get_nodes_by_membership(cluster_id))
            #cluster = network.get_nodes_by_membership(cluster_id)
       # print('heads are ', heads)
        # assign ordinary network to cluster heads using fcm
        for i, node in enumerate(sensor_nodes):
            if node in heads:
                continue
            if isinstance(node,Controller):
                continue
            cluster_id = membership[i]
            node.membership = cluster_id
            head = [x for x in heads if x.membership == cluster_id][0]
            #node.next_hop = head.id    
                    #print('next heads ', next_head.id)
            #head.next_hop = network[-1].id
            #head2 = [x for x in heads if x.membership == cluster_id][1]
            head2 = self.get_CH_neighbour(head, network[0:-1])
            
           # print('head id', head.id)
            if len(sensor_nodes)>10:
                split_list = [round(0.6 * len(sensor_nodes)), round(0.4 * len(sensor_nodes))]
                temp = iter(sensor_nodes)
                res = [list(islice(temp, 0, ele)) for ele in split_list]
            #for node in cluster:
                if node in res[0][0:]:
                    node.next_hop = head.id    
                    #print('next heads ', next_head.id)
                    head.next_hop = network[-1].id
            #for node in res[1][0:]:
                else:
                    node.next_hop = head2.id
                    head2.next_hop = network[-1].id
            else:
                node.next_hop = head.id
            #node.next_hop = head.id
            self.clusters[cluster_id] += 0

        # cluster_ids = set([x.next_hop for x in network.get_alive_nodes()])
        # print(cluster_ids)

        # for i in cluster_ids:
        #     cluster_nodes = []
        #     for node in network.get_alive_nodes():
        #         if node.next_hop == i:
        #             cluster_nodes.append(node)
        #     scheduler = SleepScheduler(cluster_nodes = cluster_nodes)
    

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

