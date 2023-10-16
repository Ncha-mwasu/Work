
import logging
import sys

import numpy as np
import skfuzzy

import config as cf
from python.network.network import Network
from python.network.node import *
from python.routing.mte import *
from python.routing.routing_protocol import *
#from python.sleep_scheduling.sleep_scheduler import *
from python.utils.utils import *


"""Every node communicate its position to the base station. Then the 
BS uses FCM to define clusters and broadcast this information to the
network. Finally, a round is executed.
"""


class FCM(RoutingProtocol):

    width = cf.AREA_WIDTH
    length = cf.AREA_LENGTH/cf.NB_CONTROLLERS
    networks = []

    def node_in_controller_region(self, node, controller):
        """ Returns the controller in the region of the node """
        # If a node is within the controller region
        return (controller.pos_x - FCM.length/2) <= node.pos_x and \
            (FCM.length/2 + controller.pos_x) > node.pos_x


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
            FCM.networks.append(controller_network)

    def setup_phase(self, network, round_nb):
        """ Network Setup Phase. The network setup for all the controller networks
        are executed
        """
        for i, ntwk in enumerate(FCM.networks):
            
            sensor_nodes = ntwk.get_sensor_nodes()
        # calculate the average distance to the BS
            #def transform(node): return calculate_distance(node, ntwk[-1])
            #distances_to_BS = [transform(node) for node in sensor_nodes]
            #avg_distance_to_BS = np.average(distances_to_BS)
           # distances_to_BS = [transform(node) for node in sensor_nodes]
           # avg_distance_to_BS = np.average(distances_to_BS)
            nb_clusters = calculate_opt_nb_clusters(len(sensor_nodes))

        # using a constant because calculating this value on-the-fly gives
        # different result than the paper
            cf.NB_CLUSTERS = nb_clusters 

        # format data to shape expected by skfuzzy API
            data = [[node.pos_x, node.pos_y] for node in ntwk[0:-1]]
            #print(data)
        #data = [[node.pos_x, node.pos_y]
        #        for node in network.get_alive_nodes()]
            data = np.array(data).transpose()
            centroids, membership = skfuzzy.cluster.cmeans(data, nb_clusters,
                                                       cf.FUZZY_M, error=0.005,
                                                       maxiter=1000,
                                                       init=None)[0:2]
        # assign node nearest to centroid as cluster head
            heads = []
        # also annotates centroids to network
            ntwk.centroids = []
            for cluster_id, centroid in enumerate(centroids):
                tmp_centroid = Node(0)
                tmp_centroid.pos_x = centroid[0]
                tmp_centroid.pos_y = centroid[1]
                ntwk.centroids.append(tmp_centroid)
                nearest_node = None
                shortest_distance = cf.INFINITY
                for node in network[0:-1]:
                    distance = calculate_distance(node, tmp_centroid)
                    if distance < shortest_distance:
                        nearest_node = node
                        shortest_distance = distance
                #nearest_node.next_hop = cf.BSID
                nearest_node.next_hop = ntwk[-1].id
                nearest_node.membership = cluster_id
                heads.append(nearest_node)

        # assign ordinary network to cluster heads using fcm
            for i, node in enumerate(ntwk[0:-1]):
                if node in heads:  # node is already a cluster head
                    continue
                cluster_id = np.argmax(membership[:, i])
                node.membership = cluster_id
                head = [x for x in heads if x.membership == cluster_id][0]
                node.next_hop = head.id
            if round_nb != 0:
                self.head_rotation(ntwk)
            controller = ntwk[-1]
        controller.transmit(destination=network.get_BS())

    # def _setup_phase(self, network):
    def head_rotation(self, network):
        logging.debug('FCM: head rotation')
        # head rotation
        # current cluster heads choose next cluster head with the most
        # residual energy and nearest to the cluster centroid
        for cluster_id in range(0, cf.NB_CLUSTERS):
            cluster = network.get_nodes_by_membership(cluster_id)
            # check if there is someone alive in this cluster
            if len(cluster) == 0:
                continue

           # for node in cluster:
            #    if node.next_hop == cf.SUBCONT0:
             #       self.current_head = node
                   # self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
              #  if node.next_hop == cf.SUBCONT1:
                    #self.no_of_times_am_head['id '+str(self.current_head.id)]+= 1
               #     self.current_head = node

            # someone is alive, find node with highest energy in the cluster
            # to be the next cluster head
            highest_energy = cf.MINUS_INFINITY
            next_head = None
            for node in cluster:
                if node.energy_source.energy > highest_energy:
                    highest_energy = node.energy_source.energy
                    next_head = node

            for node in cluster:
                if node != next_head:
                    node.next_hop = next_head.id
            #next_head.next_hop = cf.BSID
            next_head.next_hop = network[-1].id


# code temporary ommited
# def FCM_MTE_round(network, round_nb, local_traces=None, ret=None):
#  """Every node communicate its position to the base station. Then the
#  BS uses FCM to define clusters and broadcast this information to the
#  network. Finally, a round is executed.
#  """
#  setup_phase_fcm(network, round_nb)
#  heads = Network(init_network=network.get_heads()+[network.get_BS()])
#  setup_phase_mte(heads)
#  network.broadcast_next_hop()
#  network.run_round(round_nb)


# def FCM_PSO_round(network, round_nb, local_traces=None, sleep_schedulers=None):
#  """Every node communicate its position to the base station. Then the
#  BS uses FCM to define clusters and broadcast this information to the
#  network. Finally, a round is executed.
#  """
#  setup_phase_fcm(network, round_nb)
#  if round_nb == 0: # clusters do not change in FCM
#    clusters         = network.split_in_clusters()
#    sleep_schedulers = [SleepScheduler(cluster) for cluster in clusters]
#
#  for sleep_scheduler in sleep_schedulers:
#    sleep_scheduler.schedule()
#  network.run_round(round_nb)
#
#  return sleep_schedulers
