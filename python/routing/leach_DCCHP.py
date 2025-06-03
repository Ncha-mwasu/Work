import logging
import sys

import config as cf
from python.network.network import *
from python.routing.routing_protocol import *
from python.utils.utils import *

class LEACH(RoutingProtocol):
    
    width = cf.AREA_WIDTH
    length = cf.AREA_LENGTH/cf.NB_CONTROLLERS
    networks = []

    def node_in_controller_region(self, node, controller):
        """ Returns the controller in the region of the node """
        # If a node is within the controller region
        return (controller.pos_x - LEACH.length/2) <= node.pos_x and \
            (LEACH.length/2 + controller.pos_x) > node.pos_x

    def pre_communication(self, network):
        """This method is called before round 0."""
        # Setup Controllers on network
        super().pre_communication(network)

        # Nodes closest to centroids i.e
        #centroids = []

        
        for controller in network.controller_list:
            #controller.pos_x = MLC.length/2 + (MLC.length*i)
            #controller.pos_y = MLC.width/2
            
            nodes = [node for node in network.get_sensor_nodes() if \
                        self.node_in_controller_region(node, controller)]
            #for node in nodes:
             #   node.membership = controller.id            
            controller_network = Network(sensor_nodes= nodes)
            #controller_network.append(controller)
            LEACH.networks.append(controller_network)

    def setup_phase(self, network, round_nb=None):
        """The base station decides which nodes are cluster heads in this
        round, depending on a probability. Then it broadcasts this information
        to all network.
        Reference:
            W. Heinzelman, A. Chandrakasan, and H. Balakrishnan, Energy-
            efficient communication protocols for wireless sensor networks, In
            Proceedings of the 33rd Annual Hawaii International Conference on
            System Sciences (HICSS), Hawaii, USA, January 2000.
        """
        logging.info('LEACH: setup phase.')
        # decide which network are cluster heads
        for i, ntwk in enumerate(LEACH.networks): 
            sensor_nodes = ntwk.get_sensor_nodes()
            
            heads = []
            energy_list = sorted(sensor_nodes, key=lambda node: node.energy_source.energy)
            for node in sensor_nodes:
                node.sq_dist = sum([calculate_distance(node, next_node)**2 for next_node in sensor_nodes if next_node != node])
            distance_list = sorted(sensor_nodes, key=lambda node: node.sq_dist, reverse=True)
            for node in sensor_nodes:
                weight_energy = energy_list.index(node) + 1
                weight_distance = distance_list.index(node) + 1
                node.weight = weight_energy + weight_distance

            optimal_list = sorted(sensor_nodes, key=lambda node: node.weight, reverse=True)
            top_five = int(round(len(optimal_list)*0.05))
            cluster_heads = optimal_list[:top_five]
            for node in cluster_heads:
                node.next_hop = ntwk [-1].id
                heads.append(node)

        # ordinary network choose nearest cluster heads
            logging.info(
                'LEACH: ordinary nodes choose nearest nearest cluster head')
            for i, node in enumerate(ntwk[0:-1]):
                if node in heads:  # node is cluster head
                    continue
                nearest_head = heads[0]
            # find the nearest cluster head
                for head in heads[1:]:
                    if calculate_distance(node, nearest_head) > calculate_distance(node, head):
                        nearest_head = head

                node.next_hop = nearest_head.id
                controller = ntwk[-1]
                controller.transmit(destination=network.get_BS())
            ntwk.broadcast_next_hop()             
